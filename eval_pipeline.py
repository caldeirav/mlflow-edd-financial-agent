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


def _log_assessment(
    *,
    name: str,
    kind: str,
    inputs: Any,
    outputs: Any,
    expectations: Any,
    fb: Any,
    elapsed: float,
    error: Exception | None = None,
) -> None:
    ticker = ""
    question = ""
    if isinstance(inputs, dict):
        ticker = str(inputs.get("ticker") or "")
        question = str(inputs.get("question") or "")

    ct.banner(
        f"ASSESS · {name}",
        kind=kind,
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
    if error is not None:
        ct.step("fail", f"{name} raised", f"{type(error).__name__}: {error}")
        ct.outcome(False, f"{name} error in {elapsed:.1f}s")
        ct.rule("─")
        return

    value, rationale, err = ct.extract_feedback_fields(fb)
    if err:
        ct.step("fail", "Assessment error", ct.truncate(str(err), 300))
        ct.outcome(False, f"{name} error in {elapsed:.1f}s")
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
        ct.outcome(bool(positive), f"{name} → {value}  ({elapsed:.1f}s)")
    ct.rule("─")


def _wrap_scorers(scorers: list[Any]) -> list[Any]:
    """Return real MLflow Scorer instances that print console assessment traces.

    Plain wrappers are rejected by ``validate_scorers`` (must be ``isinstance(..., Scorer)``).
    """
    wrapped: list[Any] = []
    for inner in scorers:
        name = getattr(inner, "name", None) or getattr(inner, "__name__", "scorer")
        kind = "code" if name in {"RequiredMarkdownSections", "RequiredToolsUsed"} else "judge"

        def _make(inner_scorer: Any = inner, scorer_name: str = name, scorer_kind: str = kind):
            @scorer(name=scorer_name)
            def traced(
                *,
                inputs: Any = None,
                outputs: Any = None,
                expectations: dict[str, Any] | None = None,
                trace: Any = None,
            ) -> Any:
                started = time.perf_counter()
                try:
                    if hasattr(inner_scorer, "run"):
                        fb = inner_scorer.run(
                            inputs=inputs,
                            outputs=outputs,
                            expectations=expectations,
                            trace=trace,
                        )
                    else:
                        fb = inner_scorer(
                            inputs=inputs,
                            outputs=outputs,
                            expectations=expectations,
                            trace=trace,
                        )
                except Exception as exc:  # noqa: BLE001
                    _log_assessment(
                        name=scorer_name,
                        kind=scorer_kind,
                        inputs=inputs,
                        outputs=outputs,
                        expectations=expectations,
                        fb=None,
                        elapsed=time.perf_counter() - started,
                        error=exc,
                    )
                    raise
                _log_assessment(
                    name=scorer_name,
                    kind=scorer_kind,
                    inputs=inputs,
                    outputs=outputs,
                    expectations=expectations,
                    fb=fb,
                    elapsed=time.perf_counter() - started,
                )
                return fb

            return traced

        wrapped.append(_make())
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
    # Skip the pre-eval "test first sample" predict so we don't double-run case 1
    # (and so console case counters stay 1..N for the real golden suite).
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")


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


_EMPTY_REPORT_PLACEHOLDER = (
    "(empty report — no Markdown analysis was produced)"
)


def _prepare_trace_for_memalign(tr: Any) -> Any:
    """Ensure MemAlign can extract a non-empty ``outputs`` string from the trace.

    MLflow MemAlign drops traces when ``extract_response_from_trace`` is falsy
    (empty assistant content). Empty agent reports (AMZN/V) then hard-fail
    alignment even though HUMAN Groundedness feedback is present. Inject a
    placeholder into the root span outputs so empty→ungrounded can be learned.
    """
    from mlflow.entities.span import Span
    from mlflow.entities.trace import Trace
    from mlflow.entities.trace_data import TraceData
    from mlflow.genai.utils.trace_utils import extract_response_from_trace

    if extract_response_from_trace(tr):
        return tr

    new_spans: list[Any] = []
    patched = False
    for sp in getattr(tr.data, "spans", None) or []:
        d = sp.to_dict()
        is_root = getattr(sp, "parent_id", None) is None
        if is_root or getattr(sp, "name", None) == "LangGraph":
            attrs = d.get("attributes") or {}
            raw_outs = attrs.get("mlflow.spanOutputs")
            was_str = isinstance(raw_outs, str)
            try:
                outs_obj = json.loads(raw_outs) if was_str else raw_outs
            except Exception:
                outs_obj = None
            if isinstance(outs_obj, dict) and isinstance(outs_obj.get("messages"), list):
                msgs = outs_obj["messages"]
                for i in range(len(msgs) - 1, -1, -1):
                    content = msgs[i].get("content") if isinstance(msgs[i], dict) else None
                    if content == "" or content is None:
                        msgs[i]["content"] = _EMPTY_REPORT_PLACEHOLDER
                        patched = True
                        break
                if not patched and msgs and isinstance(msgs[-1], dict):
                    msgs[-1]["content"] = _EMPTY_REPORT_PLACEHOLDER
                    patched = True
                attrs["mlflow.spanOutputs"] = (
                    json.dumps(outs_obj) if was_str else outs_obj
                )
                d["attributes"] = attrs
        new_spans.append(Span.from_dict(d))

    if not patched:
        return tr

    return Trace(
        info=tr.info,
        data=TraceData(
            spans=new_spans,
            request=getattr(tr.data, "request", None),
            response=getattr(tr.data, "response", None),
        ),
    )


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
        prepared: list[Any] = []
        for tr in feedback_traces:
            fixed = _prepare_trace_for_memalign(tr)
            if fixed is not tr:
                ct.kv(
                    "memalign_empty_output_patch",
                    getattr(tr.info, "trace_id", "?"),
                )
            prepared.append(fixed)
        ct.kv(f"align_{name}_human_traces", len(prepared))
        base = judges[name]
        new_judge = base.align(traces=prepared, optimizer=optimizer)
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
    rescored = [r for r in runs if r.data.tags.get("eval_phase") == "rescored"]
    lines = [
        f"dataset_version={dataset_version}",
        f"uncalibrated_runs={len(uncal)}",
        f"aligned_runs={len(aligned)}",
        f"rescored_runs={len(rescored)}",
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


def _report_from_trace(tr: Any) -> str:
    """Best-effort Markdown report from a LangGraph eval trace."""
    spans = getattr(getattr(tr, "data", None), "spans", None) or []
    # Prefer root LangGraph outputs messages[-1]
    for span in spans:
        if getattr(span, "name", None) in {"LangGraph", "agent"}:
            outs = getattr(span, "outputs", None) or {}
            messages = outs.get("messages") if isinstance(outs, dict) else None
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    if isinstance(content, list):
                        parts = [
                            str(b.get("text", ""))
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        text = "\n".join(p for p in parts if p)
                        if text.strip():
                            return text
    # Fallback: response JSON string
    resp = getattr(getattr(tr, "data", None), "response", None)
    blob = resp if isinstance(resp, str) else json.dumps(resp, default=str)
    if "## Price context" in blob:
        idx = blob.find("## Price context")
        # stop at JSON escaping boundary if needed
        chunk = blob[idx : idx + 8000]
        chunk = chunk.replace("\\n", "\n").replace('\\"', '"')
        return chunk
    return ""


def _inputs_expectations_for_ticker(
    ticker: str, dataset_version: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    for rec in golden_dataset.build_case_records(dataset_version):
        if rec["inputs"].get("ticker") == ticker:
            return dict(rec["inputs"]), dict(rec["expectations"])
    question = config.build_analysis_question(ticker)
    return (
        {"question": question, "ticker": ticker},
        {
            "required_sections": list(config.REQUIRED_MARKDOWN_SECTIONS),
            "required_tools": list(config.REQUIRED_TOOLS_DEFAULT),
            "groundedness_policy": "live_tool_outputs",
            "dataset_version": dataset_version,
        },
    )


def _ticker_from_trace(tr: Any) -> str | None:
    req = getattr(getattr(tr, "data", None), "request", None)
    blob = req if isinstance(req, str) else json.dumps(req, default=str)
    for ticker, _ in [
        ("AAPL", None),
        ("MSFT", None),
        ("GOOGL", None),
        ("AMZN", None),
        ("NVDA", None),
        ("META", None),
        ("JPM", None),
        ("XOM", None),
        ("JNJ", None),
        ("V", None),
    ]:
        if f"of {ticker}." in blob or f"of {ticker} " in blob:
            return ticker
    return None


def rebuild_eval_row_from_trace(
    tr: Any, dataset_version: str = config.DATASET_VERSION
) -> dict[str, Any] | None:
    """Rebuild inputs/outputs/expectations from a baseline LangGraph trace."""
    ticker = _ticker_from_trace(tr)
    if not ticker:
        return None
    inputs, expectations = _inputs_expectations_for_ticker(ticker, dataset_version)
    report = _report_from_trace(tr)
    spans = getattr(getattr(tr, "data", None), "spans", None) or []
    tools_called, tool_observations = agent.tools_from_trace_spans(spans)
    outputs = {
        "report": report,
        "tools_called": tools_called,
        "tool_observations": tool_observations,
        "source_trace_id": getattr(getattr(tr, "info", None), "trace_id", None),
    }
    return {"inputs": inputs, "outputs": outputs, "expectations": expectations}


def rescore_from_run(
    run_id: str,
    *,
    dataset_version: str = config.DATASET_VERSION,
    judge_names: list[str] | None = None,
    include_code_scorers: bool = True,
) -> Any:
    """Re-run judges on existing baseline traces without re-invoking the agent.

    Rebuilds ``tool_observations`` from full tool spans (compacted, not
    blind-truncated) so Groundedness sees the same facts as the UI.
    """
    _configure_eval_runtime()
    config.init_mlflow()
    traces = mlflow.search_traces(run_id=run_id, return_type="list", max_results=200)
    # Prefer sample traces that already carry judge assessments (one per case).
    sample_traces = []
    for tr in traces:
        names = {getattr(a, "name", None) for a in (getattr(tr.info, "assessments", None) or [])}
        if "Groundedness" in names or "RequiredMarkdownSections" in names:
            sample_traces.append(tr)
    if not sample_traces:
        # Fallback: LangGraph root traces
        sample_traces = [
            tr
            for tr in traces
            if any(
                getattr(s, "name", None) == "LangGraph"
                for s in (getattr(tr.data, "spans", None) or [])
            )
        ]

    rows: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()
    for tr in sample_traces:
        row = rebuild_eval_row_from_trace(tr, dataset_version)
        if not row:
            continue
        ticker = row["inputs"].get("ticker")
        if ticker in seen_tickers:
            continue
        seen_tickers.add(str(ticker))
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No rebuildable eval rows found on run {run_id}. "
            "Pass the baseline-eval run id that logged Groundedness assessments."
        )

    judges = {j.name: j for j in build_uncalibrated_judges()}
    selected = judge_names or list(judges)
    unknown = [n for n in selected if n not in judges]
    if unknown:
        raise ValueError(f"Unknown judge(s): {unknown}")
    scorers_raw: list[Any] = []
    if include_code_scorers:
        scorers_raw.extend([RequiredMarkdownSections, RequiredToolsUsed])
    scorers_raw.extend(judges[n] for n in selected)
    scorers = _wrap_scorers(scorers_raw)

    _CASE_COUNTER["i"] = 0
    _CASE_COUNTER["n"] = len(rows)
    ct.banner(
        "RESCORE (no agent rerun)",
        source_run=run_id,
        cases=len(rows),
        judges=", ".join(selected),
        dataset_version=dataset_version,
    )
    # Sanity: news compaction should retain late articles
    for row in rows:
        if row["inputs"].get("ticker") in {"AMZN", "JNJ"}:
            blob = json.dumps(row["outputs"].get("tool_observations") or [], default=str)
            ct.kv(
                f"evidence_check_{row['inputs']['ticker']}",
                f"FedEx={('FedEx' in blob)} 25.31={('25.31' in blob)} chars={len(blob)}",
            )

    tags = config.run_tags(
        judge_version=f"{config.JUDGE_VERSION_UNCALIBRATED}-rescored",
        dataset_version=dataset_version,
        alignment_round=0,
        eval_phase="rescored",
        extra={"source_run_id": run_id, "rescore": "tool_observations_compact"},
    )
    with mlflow.start_run(run_name=f"rescore-{dataset_version}-{run_id[:8]}"):
        mlflow.set_tags(tags)
        # No predict_fn: score rebuilt outputs only.
        results = mlflow.genai.evaluate(data=rows, scorers=scorers)
    _print_eval_summary(results, phase="rescored", dataset_version=dataset_version)
    return results
