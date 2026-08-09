"""Evaluation & alignment pipeline.

Privacy: send only public-market questions, assistant Markdown outputs, and
tool-span summaries needed for judging to Gemini. No account/PII payloads.

Supports:
- Code scorers (Markdown sections, required tools)
- Gemini make_judge scorers
- Seeded human feedback (mimics MLflow UI overrides)
- MemAlign align/register and aligned re-evaluation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer
from mlflow.genai.scorers import scorer

import agent
import config
import golden_dataset

JudgeName = Literal["ToolCallEfficiency", "ToolCallCorrectness", "Groundedness"]


def _outputs_text(outputs: Any) -> str:
    if outputs is None:
        return ""
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict) and "response" in outputs:
        return str(outputs["response"])
    return str(outputs)


def _tool_names_from_trace(trace: mlflow.entities.Trace | None) -> set[str]:
    names: set[str] = set()
    if trace is None:
        return names
    try:
        for span in trace.data.spans:
            span_type = getattr(span, "span_type", None) or ""
            name = getattr(span, "name", "") or ""
            if "TOOL" in str(span_type).upper() or name in config.REQUIRED_TOOLS_DEFAULT:
                # Prefer original tool name when present in attributes
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
    expectations: dict[str, Any] | None = None,
    trace: mlflow.entities.Trace | None = None,
    **_: Any,
) -> Feedback:
    required = set((expectations or {}).get("required_tools") or config.REQUIRED_TOOLS_DEFAULT)
    used = _tool_names_from_trace(trace)
    # Match if required tool name appears as substring of any span name (adapter prefixes)
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
    efficiency = make_judge(
        name="ToolCallEfficiency",
        instructions=(
            "Analyze {{ trace }} for redundant tool calls or reasoning thrash.\n"
            "Return 'efficient' if tool use is lean and purposeful; "
            "'inefficient' if there are clear redundant or thrashing calls."
        ),
        model=config.JUDGE_MODEL,
        feedback_value_type=Literal["efficient", "inefficient"],
    )
    correctness = make_judge(
        name="ToolCallCorrectness",
        instructions=(
            "Analyze {{ trace }} and {{ expectations }}.\n"
            "Decide if the agent selected appropriate tools and arguments for the "
            "user request. Return 'correct' or 'incorrect'."
        ),
        model=config.JUDGE_MODEL,
        feedback_value_type=Literal["correct", "incorrect"],
    )
    groundedness = make_judge(
        name="Groundedness",
        instructions=(
            "Compare {{ outputs }} to tool results in {{ trace }}.\n"
            "Ground truth is live tool output on this trace (not frozen snapshots).\n"
            "Return 'grounded' if numeric/factual claims are supported by tools; "
            "'ungrounded' if the answer invents or contradicts tool data."
        ),
        model=config.JUDGE_MODEL,
        feedback_value_type=Literal["grounded", "ungrounded"],
    )
    return [efficiency, correctness, groundedness]


def _predict_fn(**inputs: Any) -> str:
    question = inputs.get("question") or ""
    ticker = inputs.get("ticker")
    return agent.predict_fn(question=question, ticker=ticker)


def run_baseline_eval(dataset_version: str = config.DATASET_VERSION) -> Any:
    """Uncalibrated evaluate over golden dataset."""
    experiment_id = config.init_mlflow()
    golden_dataset.ensure_golden_dataset(dataset_version, experiment_id=experiment_id)
    data = golden_dataset.eval_dataframe_records(dataset_version)
    scorers = [RequiredMarkdownSections, RequiredToolsUsed, *build_uncalibrated_judges()]
    tags = config.run_tags(
        judge_version=config.JUDGE_VERSION_UNCALIBRATED,
        dataset_version=dataset_version,
        alignment_round=0,
        eval_phase="uncalibrated",
    )
    with mlflow.start_run(run_name=f"baseline-eval-{dataset_version}"):
        mlflow.set_tags(tags)
        results = mlflow.genai.evaluate(data=data, predict_fn=_predict_fn, scorers=scorers)
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
            # AssessmentSourceType may compare as enum
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
            # register may be optional depending on MLflow version
            pass
        aligned[name] = new_judge
    return aligned


def run_aligned_eval(
    aligned_judges: dict[str, Any],
    dataset_version: str = config.DATASET_VERSION,
    alignment_round: int = 1,
) -> Any:
    """Re-evaluate golden dataset with aligned judges (baseline evidence retained)."""
    config.init_mlflow()
    data = golden_dataset.eval_dataframe_records(dataset_version)
    # Keep code scorers; replace qualitative judges that were aligned
    base = {j.name: j for j in build_uncalibrated_judges()}
    base.update(aligned_judges)
    scorers = [RequiredMarkdownSections, RequiredToolsUsed, *base.values()]
    judge_version = f"{config.JUDGE_VERSION_ALIGNED_PREFIX}-{alignment_round}"
    tags = config.run_tags(
        judge_version=judge_version,
        dataset_version=dataset_version,
        alignment_round=alignment_round,
        eval_phase="aligned",
    )
    with mlflow.start_run(run_name=f"aligned-eval-{dataset_version}-r{alignment_round}"):
        mlflow.set_tags(tags)
        results = mlflow.genai.evaluate(data=data, predict_fn=_predict_fn, scorers=scorers)
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
