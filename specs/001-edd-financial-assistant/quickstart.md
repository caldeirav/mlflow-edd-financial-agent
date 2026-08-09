# Quickstart: EDD Financial Assistant

**Feature**: `001-edd-financial-assistant` | **Date**: 2026-08-09

## Prerequisites

1. Python >=3.13, project deps installed (`uv sync`)
2. [LMStudio](https://lmstudio.ai) serving `qwen/qwen3.6-35b-a3b` at
   `http://localhost:1234/v1`
3. `GEMINI_API_KEY` in `.env` (judges, MemAlign reflection, and default
   `gemini:/text-embedding-004` embeddings) — start from `cp .env.example .env`

## Run flow

### 1. Single analysis (P1)

```bash
uv run python main.py run-agent --ticker AAPL
```

Expect Markdown with sections: Price context, News, Financial statements,
Risks/limitations.

### 2. Baseline golden eval (P2)

```bash
uv run python main.py run-baseline-eval --dataset-version v1
```

Creates/uses 10-case dataset; writes traces + uncalibrated Gemini scores to
`sqlite:///mlflow.db`.

### 3. Review / annotate (P3 — primary)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open baseline traces; add ≥5 human assessments on the judge(s) you will align
(name must match judge, e.g. `ToolCallEfficiency`).

### 3b. Demo seed (optional)

```bash
uv run python main.py seed-feedback --file data/expert_feedback_seed.json
```

### 4. Align + re-evaluate

```bash
uv run python main.py align-and-reeval --judges ToolCallEfficiency --alignment-round 1
```

Compare `eval_phase=uncalibrated` vs `eval_phase=aligned` in the UI (same
`dataset_version`).

## Module map

| File | Role |
|------|------|
| `mcp_server.py` | FastMCP yfinance tools |
| `agent.py` | LangGraph ReAct + LMStudio + autolog |
| `golden_dataset.py` | Versioned MLflow eval dataset |
| `eval_pipeline.py` | Judges, MemAlign, evaluate |
| `main.py` | Orchestration |
| `config.py` | Pins and tags |

## Growing the suite

Bump `DATASET_VERSION` / cases in `golden_dataset.py` when feedback identifies
gaps; keep prior versions used as frozen baselines intact.

## Privacy note

Only public market prompts/outputs/tool summaries are sent to Gemini.
