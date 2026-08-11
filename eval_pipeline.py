"""Evaluation & alignment pipeline.

Privacy: send only public-market questions, assistant Markdown outputs, and
tool-span summaries needed for judging to Gemini. No account/PII payloads.

Supports:
- Code scorers (Markdown sections, required tools)
- Gemini make_judge scorers (inputs/outputs only — no {{ trace }}; Gemini rejects
  function-calling + application/json together)
- Seeded human feedback (mimics MLflow UI overrides)
- MemAlign align/register and aligned re-evaluation
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer
from mlflow.genai.scorers import scorer

import agent
import config
import console_trace as ct
import golden_dataset

JudgeName = Literal["ToolCallEfficiency", "ToolCallCorrectness", "Groundedness"]

_CASE_COUNTER = {"i": 0, "n": 0}


def _outputs_text(outputs: Any) -> str:
    if outputs is None:
        return ""
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        if "report" in outputs:
            return str(outputs["report"])
        if "response" in outputs:
            return str(outputs["response"])
    return str(outputs)


def _tools_from_outputs(outputs: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(outputs, dict):
        return names
    for item in outputs.get("tools_called") or []:
        names.add(str(item))
    for obs in outputs.get("tool_observations") or []:
        if isinstance(obs, dict) and obs.get("name"):
            names.add(str(obs["name"]))
    return names


def _tool_names_from_trace(trace: mlflow.entities.Trace | None) -> set[str]:
    names: set[str] = set()
    if trace is None:
        return names
    try:
        for span in trace.data.spans:
            span_type = getattr(span, "span_type", None) or ""
            name = getattr(span, "name", "") or ""
            if "TOOL" in str(span_type).upper() or name in config.REQUIRED_TOOLS_DEFAULT:
                names.add(name)
                attrs = getattr(span, "attributes", None) or {}
                for key in ("tool_name", "function_name", "name"):
                    if key in attrs:
                        names.add(str(attrs[key]))
    except Exception:
        pass
    return names


@scorer
def RequiredMarkdownSections(
    *,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    **_: Any,
) -> Feedback:
    text = _outputs_text(outputs)
    required = list((expectations or {}).get("required_sections") or config.REQUIRED_MARKDOWN_SECTIONS)
    missing = [h for h in required if h not in text]
    ok = len(missing) == 0
    return Feedback(
        value=ok,
        rationale="All required sections present"
        if ok
        else f"Missing Markdown sections: {missing}",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="RequiredMarkdownSections"),
    )


@scorer
def RequiredToolsUsed(
    *,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: mlflow.entities.Trace | None = None,
    **_: Any,
) -> Feedback:
    required = set((expectations or {}).get("required_tools") or config.REQUIRED_TOOLS_DEFAULT)
    used = _tool_names_from_trace(trace) | _tools_from_outputs(outputs)
    missing = []
    for tool in required:
        if not any(tool in u for u in used):
            missing.append(tool)
    ok = len(missing) == 0
    return Feedback(
        value=ok,
        rationale="All required tools used"
        if ok
        else f"Missing required tools: {missing}; seen={sorted(used)}",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="RequiredToolsUsed"),
    )


def build_uncalibrated_judges() -> list[Any]:
    # IMPORTANT: do not use {{ trace }} with Gemini judges.
    # MLflow's trace agent uses function calling + response_mime_type=application/json,
    # which Gemini rejects. Pass tool summaries via {{ outputs }} instead.
    efficiency = make_judge(
        name="ToolCallEfficiency",
        instructions=(
            "You are evaluating a financial-analysis agent.\n"
            "Inputs: {{ inputs }}\n"
            "Outputs (report + tools_called + tool_observations): {{ outputs }}\n"
            "Return 'efficient' if tool use looks lean and purposeful "
            "(needed tools called without obvious redundant thrash); "
            "'inefficient' if there are clear redundant/repeated calls or thrashing."
        ),
        model=config.JUDGE_MODEL,
        feedback_value_type=Literal["efficient", "inefficient"],
    )
    correctness = make_judge(
        name="ToolCallCorrectness",
        instructions=(
            "You are evaluating a financial-analysis agent.\n"
            "Inputs: {{ inputs }}\n"
            "Expectations: {{ expectations }}\n"
            "Outputs (report + tools_called + tool_observations): {{ outputs }}\n"
            "Decide if the agent selected appropriate tools for the request "
            "(and covered expected required_tools when present). "
            "Return 'correct' or 'incorrect'."
        ),
        model=config.JUDGE_MODEL,
        feedback_value_type=Literal["correct", "incorrect"],
    )
    groundedness = make_judge(
        name="Groundedness",
        instructions=(
            "You are evaluating a financial-analysis agent.\n"
            "Inputs: {{ inputs }}\n"
            "Outputs include the Markdown report and tool_observations "
            "(live tool results from this run — not frozen snapshots): {{ outputs }}\n"
            "Return 'grounded' if numeric/factual claims in the report are supported "
            "by tool_observations; 'ungrounded' if the answer invents or contradicts "
            "tool data. If tools failed and the report says unavailable, that can still "
            "be grounded."
        ),
        model=config.JUDGE_MODEL,
        feedback_value_type=Literal["grounded", "ungrounded"],
    )
    return [efficiency, correctness, groundedness]


class _ConsoleScorer:
    """Wrap an MLflow scorer/judge with readable stdio assessment traces."""

    def __init__(self, inner: Any, *, kind: str = "judge") -> None:
        self._inner = inner
        self._kind = kind
        self.name = getattr(inner, "name", None) or getattr(inner, "__name__", "scorer")

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._trace_call(args, kwargs)

    def run(self, **kwargs: Any) -> Any:
        return self._trace_call((), kwargs)

    def _trace_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        inputs = kwargs.get("inputs") if "inputs" in kwargs else (args[0] if args else {})
        outputs = kwargs.get("outputs")
        expectations = kwargs.get("expectations")
        if outputs is None and len(args) > 1:
            outputs = args[1]

        ticker = ""
        question = ""
        if isinstance(inputs, dict):
            ticker = str(inputs.get("ticker") or "")
            question = str(inputs.get("question") or "")

        ct.banner(
            f"ASSESS · {self.name}",
            kind=self._kind,
            ticker=ticker or None,
            case=f"{_CASE_COUNTER['i']}/{_CASE_COUNTER['n']}"
            if _CASE_COUNTER["n"]
            else None,
        )
        ct.section("Subject under assessment")
        if question:
            ct.block("question", question, limit=280)
        tools = sorted(_tools_from_outputs(outputs))
        ct.kv("tools_called", ", ".join(tools) if tools else ct.dim("(none)"))
        if expectations:
            ct.kv("expectations", ct.truncate(json.dumps(expectations, default=str), 220))
        ct.block("report", _outputs_text(outputs), limit=420)

        ct.section("Judge / scorer")
        ct.step(self._kind, f"Invoking {self.name}…")
        started = time.perf_counter()
        try:
            if args:
                fb = self._inner(*args, **kwargs)
            elif hasattr(self._inner, "run"):
                fb = self._inner.run(**kwargs)
            else:
                fb = self._inner(**kwargs)
        except Exception as exc:  # noqa: BLE001
            ct.step("fail", f"{self.name} raised", f"{type(exc).__name__}: {exc}")
            raise
        elapsed = time.perf_counter() - started
        value, rationale, err = ct.extract_feedback_fields(fb)
        if err:
            ct.step("fail", "Assessment error", ct.truncate(str(err), 300))
            ct.outcome(False, f"{self.name} error in {elapsed:.1f}s")
        else:
            ct.step("ok", f"Assessment value: {ct.feedback_value(value)}")
            if rationale:
                ct.block("rationale", str(rationale), limit=360)
            positive = str(value).lower() in {
                "true",
                "efficient",
                "correct",
                "grounded",
                "pass",
            } or value is True
            ct.outcome(bool(positive), f"{self.name} → {value}  ({elapsed:.1f}s)")
        ct.rule("─")
        return fb


def _wrap_scorers(scorers: list[Any]) -> list[Any]:
    wrapped: list[Any] = []
    for s in scorers:
        name = getattr(s, "name", None) or getattr(s, "__name__", "")
        kind = "code" if name in {"RequiredMarkdownSections", "RequiredToolsUsed"} else "judge"
        wrapped.append(_ConsoleScorer(s, kind=kind))
    return wrapped


def _predict_fn(**inputs: Any) -> dict[str, Any]:
    _CASE_COUNTER["i"] += 1
    question = inputs.get("question") or ""
    ticker = inputs.get("ticker")
    ct.banner(
        f"EVAL CASE {_CASE_COUNTER['i']}/{_CASE_COUNTER['n'] or '?'}",
        ticker=ticker,
        phase="predict",
    )
    if question:
        ct.block("question", question, limit=280)
    return agent.predict_fn(question=question, ticker=ticker)


def _configure_eval_runtime() -> None:
    # Parallel predict_fn workers fight over MCP stdio + nested asyncio.run.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")


def _print_eval_summary(results: Any, *, phase: str, dataset_version: str) -> None:
    ct.banner("EVAL SUMMARY", phase=phase, dataset_version=dataset_version)
    metrics = getattr(results, "metrics", None) or {}
    if metrics:
        ct.section("Aggregate metrics")
        for key in sorted(metrics):
            ct.kv(key, metrics[key])
    run_id = getattr(results, "run_id", None)
    if run_id:
        ct.kv("run_id", run_id)
    ct.rule("═")


def run_baseline_eval(dataset_version: str = config.DATASET_VERSION) -> Any:
    """Uncalibrated evaluate over golden dataset."""
    _configure_eval_runtime()
    experiment_id = config.init_mlflow()
    golden_dataset.ensure_golden_dataset(dataset_version, experiment_id=experiment_id)
    data = golden_dataset.eval_dataframe_records(dataset_version)
    _CASE_COUNTER["i"] = 0
    _CASE_COUNTER["n"] = len(data)
    scorers = _wrap_scorers(
        [RequiredMarkdownSections, RequiredToolsUsed, *build_uncalibrated_judges()]
    )
    tags = config.run_tags(
        judge_version=config.JUDGE_VERSION_UNCALIBRATED,
        dataset_version=dataset_version,
        alignment_round=0,
        eval_phase="uncalibrated",
    )
    ct.banner(
        "BASELINE EVAL",
        dataset_version=dataset_version,
        cases=_CASE_COUNTER["n"],
        judge_model=config.JUDGE_MODEL,
    )
    with mlflow.start_run(run_name=f"baseline-eval-{dataset_version}"):
        mlflow.set_tags(tags)
        results = mlflow.genai.evaluate(data=data, predict_fn=_predict_fn, scorers=scorers)
    _print_eval_summary(results, phase="uncalibrated", dataset_version=dataset_version)
    return results


def seed_human_feedback(seed_path: str | Path) -> int:
    """Attach HUMAN assessments from a seed JSON file onto matching traces.

    Seed schema: list of
    {case_id?, trace_id?, judge_name, value, rationale}
    If trace_id missing, match latest trace whose request/inputs contain case_id/ticker.
    """
    config.init_mlflow()
    path = Path(seed_path)
    items = json.loads(path.read_text(encoding="utf-8"))
    traces = mlflow.search_traces(return_type="list", max_results=200)
    applied = 0
    human = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="expert_seed"
    )
    for item in items:
        trace_id = item.get("trace_id")
        if not trace_id:
            needle = item.get("case_id") or item.get("ticker") or ""
            for tr in traces:
                blob = ""
                try:
                    blob = json.dumps(tr.data.request or {}) + json.dumps(
                        [s.inputs for s in tr.data.spans[:3]], default=str
                    )
                except Exception:
                    blob = str(tr)
                if needle and needle in blob:
                    trace_id = tr.info.trace_id
                    break
        if not trace_id:
            continue
        mlflow.log_feedback(
            trace_id=trace_id,
            name=item["judge_name"],
            value=item["value"],
            rationale=item.get("rationale") or "Expert override",
            source=human,
        )
        applied += 1
    return applied


def _traces_with_human_feedback(judge_name: str) -> list[Any]:
    traces = mlflow.search_traces(return_type="list", max_results=200)
    selected = []
    for tr in traces:
        assessments = getattr(tr.info, "assessments", None) or []
        for fb in assessments:
            name = getattr(fb, "name", None)
            source = getattr(fb, "source", None)
            source_type = getattr(source, "source_type", None) if source else None
            if name == judge_name and str(source_type).endswith("HUMAN"):
                selected.append(tr)
                break
            if name == judge_name and source_type == AssessmentSourceType.HUMAN:
                selected.append(tr)
                break
    return selected


def align_judges(
    judge_names: list[str],
    alignment_round: int = 1,
) -> dict[str, Any]:
    """MemAlign selected judges using HUMAN assessments on traces."""
    config.init_mlflow()
    experiment_id = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id
    judges = {j.name: j for j in build_uncalibrated_judges()}
    optimizer = MemAlignOptimizer(
        reflection_lm=config.MEMALIGN_REFLECTION_LM,
        embedding_model=config.MEMALIGN_EMBEDDING_MODEL,
    )
    aligned: dict[str, Any] = {}
    for name in judge_names:
        if name not in judges:
            raise ValueError(f"Unknown judge: {name}")
        feedback_traces = _traces_with_human_feedback(name)
        if len(feedback_traces) < 1:
            raise RuntimeError(
                f"No HUMAN assessments found for judge '{name}'. "
                "Annotate in MLflow UI or run seed-feedback first."
            )
        base = judges[name]
        new_judge = base.align(traces=feedback_traces, optimizer=optimizer)
        try:
            new_judge.register(experiment_id=experiment_id)
        except Exception:
            pass
        aligned[name] = new_judge
    return aligned


def run_aligned_eval(
    aligned_judges: dict[str, Any],
    dataset_version: str = config.DATASET_VERSION,
    alignment_round: int = 1,
) -> Any:
    """Re-evaluate golden dataset with aligned judges (baseline evidence retained)."""
    _configure_eval_runtime()
    config.init_mlflow()
    data = golden_dataset.eval_dataframe_records(dataset_version)
    _CASE_COUNTER["i"] = 0
    _CASE_COUNTER["n"] = len(data)
    base = {j.name: j for j in build_uncalibrated_judges()}
    base.update(aligned_judges)
    scorers = _wrap_scorers(
        [RequiredMarkdownSections, RequiredToolsUsed, *base.values()]
    )
    judge_version = f"{config.JUDGE_VERSION_ALIGNED_PREFIX}-{alignment_round}"
    tags = config.run_tags(
        judge_version=judge_version,
        dataset_version=dataset_version,
        alignment_round=alignment_round,
        eval_phase="aligned",
    )
    ct.banner(
        "ALIGNED EVAL",
        dataset_version=dataset_version,
        alignment_round=alignment_round,
        cases=_CASE_COUNTER["n"],
        aligned_judges=", ".join(aligned_judges),
    )
    with mlflow.start_run(run_name=f"aligned-eval-{dataset_version}-r{alignment_round}"):
        mlflow.set_tags(tags)
        results = mlflow.genai.evaluate(data=data, predict_fn=_predict_fn, scorers=scorers)
    _print_eval_summary(
        results, phase=f"aligned-r{alignment_round}", dataset_version=dataset_version
    )
    return results


def compare_eval_phases(dataset_version: str = config.DATASET_VERSION) -> str:
    """Soft comparison helper: summarize runs tagged uncalibrated vs aligned."""
    config.init_mlflow()
    exp = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        return "No experiment found."
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.dataset_version = '{dataset_version}'",
        output_format="list",
    )
    uncal = [r for r in runs if r.data.tags.get("eval_phase") == "uncalibrated"]
    aligned = [r for r in runs if r.data.tags.get("eval_phase") == "aligned"]
    lines = [
        f"dataset_version={dataset_version}",
        f"uncalibrated_runs={len(uncal)}",
        f"aligned_runs={len(aligned)}",
    ]
    if not uncal:
        lines.append("SOFT GATE: missing uncalibrated baseline evidence.")
    if not aligned:
        lines.append("SOFT GATE: missing aligned evaluation evidence.")
    if uncal and aligned:
        lines.append(
            "OK: both uncalibrated and aligned evidence present for side-by-side comparison."
        )
    return "\n".join(lines)
