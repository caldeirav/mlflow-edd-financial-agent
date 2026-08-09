"""End-to-end orchestration for the EDD financial assistant demo.

Commands:
  run-agent | run-baseline-eval | seed-feedback | align-and-reeval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import agent
import config
import eval_pipeline
import golden_dataset


def cmd_run_agent(args: argparse.Namespace) -> None:
    config.init_mlflow()
    ticker = args.ticker or "AAPL"
    question = args.question or config.build_analysis_question(ticker, args.focus)
    print(f"Running agent for ticker={ticker} …")
    print(f"Tracking: {config.TRACKING_URI} experiment={config.EXPERIMENT_NAME}")
    output = agent.run_analysis(question)
    print(output)


def cmd_run_baseline_eval(args: argparse.Namespace) -> None:
    version = args.dataset_version or config.DATASET_VERSION
    print(f"Baseline eval dataset_version={version} …")
    results = eval_pipeline.run_baseline_eval(dataset_version=version)
    print(results)
    print(eval_pipeline.compare_eval_phases(version))


def cmd_seed_feedback(args: argparse.Namespace) -> None:
    path = Path(args.file)
    n = eval_pipeline.seed_human_feedback(path)
    print(f"Applied {n} human override assessment(s) from {path}")


def cmd_align_and_reeval(args: argparse.Namespace) -> None:
    judges = args.judges or ["ToolCallEfficiency"]
    round_n = args.alignment_round
    version = args.dataset_version or config.DATASET_VERSION
    print(f"Aligning judges={judges} round={round_n} …")
    aligned = eval_pipeline.align_judges(judges, alignment_round=round_n)
    print(f"Aligned: {list(aligned)}")
    print("Re-evaluating with aligned judges …")
    results = eval_pipeline.run_aligned_eval(
        aligned, dataset_version=version, alignment_round=round_n
    )
    print(results)
    print(eval_pipeline.compare_eval_phases(version))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLflow EDD financial assistant orchestration"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_agent = sub.add_parser("run-agent", help="Single financial analysis request")
    p_agent.add_argument("--ticker", default="AAPL")
    p_agent.add_argument("--question", default=None)
    p_agent.add_argument("--focus", default=None)
    p_agent.set_defaults(func=cmd_run_agent)

    p_base = sub.add_parser("run-baseline-eval", help="10-case uncalibrated golden eval")
    p_base.add_argument("--dataset-version", default=config.DATASET_VERSION)
    p_base.set_defaults(func=cmd_run_baseline_eval)

    p_seed = sub.add_parser(
        "seed-feedback",
        help="Attach demo HUMAN overrides (mimics MLflow UI annotations)",
    )
    p_seed.add_argument(
        "--file",
        default=str(Path("data/expert_feedback_seed.json")),
    )
    p_seed.set_defaults(func=cmd_seed_feedback)

    p_align = sub.add_parser(
        "align-and-reeval",
        help="MemAlign selected judges and re-evaluate (keeps baseline)",
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
