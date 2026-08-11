"""End-to-end orchestration for the EDD financial assistant demo.

Commands:
  run-agent | run-baseline-eval | seed-feedback | rescore-eval | align-and-reeval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import agent
import config
import console_trace as ct
import eval_pipeline


def cmd_run_agent(args: argparse.Namespace) -> None:
    if args.quiet:
        ct.set_enabled(False)
    config.init_mlflow()
    ticker = args.ticker or "AAPL"
    question = args.question or config.build_analysis_question(ticker, args.focus)
    if not ct.enabled():
        print(f"Running agent for ticker={ticker} …")
        print(f"Tracking: {config.TRACKING_URI} experiment={config.EXPERIMENT_NAME}")
    else:
        ct.kv("tracking", config.TRACKING_URI)
        ct.kv("experiment", config.EXPERIMENT_NAME)
    output = agent.run_analysis(question)
    if not ct.enabled():
        print(output.get("report", output))
        tools = output.get("tools_called") or []
        if tools:
            print(f"\n[tools_called: {', '.join(tools)}]")


def cmd_run_baseline_eval(args: argparse.Namespace) -> None:
    if args.quiet:
        ct.set_enabled(False)
    version = args.dataset_version or config.DATASET_VERSION
    if not ct.enabled():
        print(f"Baseline eval dataset_version={version} …")
    results = eval_pipeline.run_baseline_eval(dataset_version=version)
    if not ct.enabled():
        print(results)
    print(eval_pipeline.compare_eval_phases(version))


def cmd_seed_feedback(args: argparse.Namespace) -> None:
    path = Path(args.file)
    n = eval_pipeline.seed_human_feedback(path)
    print(f"Applied {n} human override assessment(s) from {path}")


def cmd_rescore_eval(args: argparse.Namespace) -> None:
    if args.quiet:
        ct.set_enabled(False)
    version = args.dataset_version or config.DATASET_VERSION
    judges = args.judges or [
        "Groundedness",
        "ToolCallCorrectness",
        "ToolCallEfficiency",
    ]
    results = eval_pipeline.rescore_from_run(
        args.run_id,
        dataset_version=version,
        judge_names=judges,
        include_code_scorers=not args.judges_only,
    )
    if not ct.enabled():
        print(results)
    print(eval_pipeline.compare_eval_phases(version))


def cmd_align_and_reeval(args: argparse.Namespace) -> None:
    if args.quiet:
        ct.set_enabled(False)
    judges = args.judges or ["ToolCallEfficiency"]
    round_n = args.alignment_round
    version = args.dataset_version or config.DATASET_VERSION
    if not ct.enabled():
        print(f"Aligning judges={judges} round={round_n} …")
    else:
        ct.banner("ALIGN", judges=", ".join(judges), round=round_n)
    aligned = eval_pipeline.align_judges(judges, alignment_round=round_n)
    print(f"Aligned: {list(aligned)}")
    if not ct.enabled():
        print("Re-evaluating with aligned judges …")
    results = eval_pipeline.run_aligned_eval(
        aligned, dataset_version=version, alignment_round=round_n
    )
    if not ct.enabled():
        print(results)
    print(eval_pipeline.compare_eval_phases(version))


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--quiet",
        action="store_true",
        help="Disable pretty stdio execution traces (or set EDD_QUIET=1)",
    )
    parser = argparse.ArgumentParser(
        description="MLflow EDD financial assistant orchestration",
        parents=[parent],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_agent = sub.add_parser(
        "run-agent", help="Single financial analysis request", parents=[parent]
    )
    p_agent.add_argument("--ticker", default="AAPL")
    p_agent.add_argument("--question", default=None)
    p_agent.add_argument("--focus", default=None)
    p_agent.set_defaults(func=cmd_run_agent)

    p_base = sub.add_parser(
        "run-baseline-eval",
        help="10-case uncalibrated golden eval",
        parents=[parent],
    )
    p_base.add_argument("--dataset-version", default=config.DATASET_VERSION)
    p_base.set_defaults(func=cmd_run_baseline_eval)

    p_seed = sub.add_parser(
        "seed-feedback",
        help="Attach demo HUMAN overrides (mimics MLflow UI annotations)",
        parents=[parent],
    )
    p_seed.add_argument(
        "--file",
        default=str(Path("data/expert_feedback_seed.json")),
    )
    p_seed.set_defaults(func=cmd_seed_feedback)

    p_rescore = sub.add_parser(
        "rescore-eval",
        help="Re-run judges on an existing baseline run (no agent rerun)",
        parents=[parent],
    )
    p_rescore.add_argument(
        "--run-id",
        required=True,
        help="Baseline eval MLflow run id (e.g. from baseline-eval-v1)",
    )
    p_rescore.add_argument("--dataset-version", default=config.DATASET_VERSION)
    p_rescore.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="Judge subset (default: all three qualitative judges)",
    )
    p_rescore.add_argument(
        "--judges-only",
        action="store_true",
        help="Skip RequiredMarkdownSections / RequiredToolsUsed code scorers",
    )
    p_rescore.set_defaults(func=cmd_rescore_eval)

    p_align = sub.add_parser(
        "align-and-reeval",
        help="MemAlign selected judges and re-evaluate (keeps baseline)",
        parents=[parent],
    )
    p_align.add_argument(
        "--judges",
        nargs="+",
        default=["ToolCallEfficiency"],
        help="Operator-selected judge name(s) for this alignment round",
    )
    p_align.add_argument("--alignment-round", type=int, default=1)
    p_align.add_argument("--dataset-version", default=config.DATASET_VERSION)
    p_align.set_defaults(func=cmd_align_and_reeval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
