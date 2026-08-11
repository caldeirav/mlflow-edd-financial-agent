"""Versioned MLflow GenAI golden evaluation dataset for financial analysis.

Bump DATASET_VERSION (and recreate/merge records) when failures or accepted
human feedback identify gaps in the suite. Do not silently overwrite prior
dataset versions used as frozen baselines.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.genai.datasets

import config

# 10 cases: same canonical request pattern, varied tickers / light focus
_GOLDEN_TICKERS: list[tuple[str, str | None]] = [
    ("AAPL", None),
    ("MSFT", "recent product and cloud news"),
    ("GOOGL", "advertising and cloud signals"),
    ("AMZN", "retail vs AWS narrative"),
    ("NVDA", "AI demand and margins"),
    ("META", "ads recovery and Reality Labs"),
    ("JPM", "net interest income focus"),
    ("XOM", "energy price sensitivity"),
    ("JNJ", "pharma pipeline and litigation risk"),
    ("V", "payments volume trends"),
]


def required_sections() -> list[str]:
    return list(config.REQUIRED_MARKDOWN_SECTIONS)


def build_case_records(dataset_version: str = config.DATASET_VERSION) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, (ticker, focus) in enumerate(_GOLDEN_TICKERS, start=1):
        question = config.build_analysis_question(ticker, focus)
        records.append(
            {
                "inputs": {
                    "question": question,
                    "ticker": ticker,
                    "case_id": f"case-{idx:02d}",
                },
                "expectations": {
                    "required_sections": required_sections(),
                    "required_tools": list(config.REQUIRED_TOOLS_DEFAULT),
                    "groundedness_policy": "live_tool_outputs",
                    "dataset_version": dataset_version,
                },
            }
        )
    return records


def ensure_golden_dataset(
    dataset_version: str = config.DATASET_VERSION,
    experiment_id: str | None = None,
):
    """Create or merge the golden dataset for the given version tag."""
    name = f"{config.DATASET_NAME}_{dataset_version}"
    records = build_case_records(dataset_version)
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=name)
        dataset = dataset.merge_records(records)
    except Exception:
        # Dataset may not exist yet
        kwargs: dict[str, Any] = {
            "name": name,
            "tags": {
                "dataset_version": dataset_version,
                "domain": "financial_analysis",
                "groundedness_policy": "live_tool_outputs",
            },
        }
        if experiment_id:
            kwargs["experiment_id"] = experiment_id
        dataset = mlflow.genai.datasets.create_dataset(**kwargs)
        dataset = dataset.merge_records(records)
    return dataset


def eval_dataframe_records(dataset_version: str = config.DATASET_VERSION) -> list[dict[str, Any]]:
    """Records suitable for mlflow.genai.evaluate(data=...)."""
    return build_case_records(dataset_version)
