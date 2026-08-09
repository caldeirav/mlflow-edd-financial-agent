"""Shared configuration pins for the EDD financial assistant demo.

Privacy: only public-market prompts, assistant Markdown, and tool summaries
may be sent to Gemini (judge / MemAlign reflection). No account or PII data.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow

# --- Tracking ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "edd-financial-assistant")

# --- Agent (LMStudio / Qwen) ---
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3.6-35b-a3b")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

# --- Judges (Gemini via MLflow provider URI) ---
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini:/gemini-2.5-pro")

# --- MemAlign pins (explicit; never rely on silent OpenAI embedding default) ---
# reflection_lm: cheaper Gemini flash-class for guideline distillation
MEMALIGN_REFLECTION_LM = os.getenv(
    "MEMALIGN_REFLECTION_LM", "gemini:/gemini-2.5-flash"
)
# embedding_model: explicitly pinned for episodic memory (Complexity Tracking:
# OpenAI embedding URI used when Gemini embeddings are unsupported by MemAlign)
MEMALIGN_EMBEDDING_MODEL = os.getenv(
    "MEMALIGN_EMBEDDING_MODEL", "openai:/text-embedding-3-small"
)

# --- Version tags ---
AGENT_VERSION = os.getenv("AGENT_VERSION", "0.1.0")
DATASET_NAME = "financial_analysis_golden"
DATASET_VERSION = os.getenv("DATASET_VERSION", "v1")
JUDGE_VERSION_UNCALIBRATED = "uncalibrated-0"
JUDGE_VERSION_ALIGNED_PREFIX = "aligned"

REQUIRED_MARKDOWN_SECTIONS = [
    "Price context",
    "News",
    "Financial statements",
    "Risks/limitations",
]

REQUIRED_TOOLS_DEFAULT = [
    "get_stock_price",
    "get_financial_news",
    "get_financial_statements",
]

CANONICAL_ANALYSIS_TEMPLATE = (
    "Provide a financial analysis of {ticker}. Cover recent price context, "
    "relevant news, key financial-statement signals, and risks/limitations. "
    "Use market-data tools for facts. Respond in Markdown with exactly these "
    "section headings: Price context, News, Financial statements, Risks/limitations."
)


def init_mlflow() -> str:
    """Configure tracking URI and experiment. Returns experiment_id."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment.experiment_id


def run_tags(
    *,
    judge_version: str,
    dataset_version: str = DATASET_VERSION,
    alignment_round: int = 0,
    eval_phase: str = "uncalibrated",
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Standard experiment/run tags for EDD attribution."""
    tags = {
        "agent_version": AGENT_VERSION,
        "judge_version": judge_version,
        "dataset_version": dataset_version,
        "alignment_round": str(alignment_round),
        "eval_phase": eval_phase,
    }
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def build_analysis_question(ticker: str, focus: str | None = None) -> str:
    question = CANONICAL_ANALYSIS_TEMPLATE.format(ticker=ticker.upper())
    if focus:
        question += f" Emphasize: {focus}."
    return question
