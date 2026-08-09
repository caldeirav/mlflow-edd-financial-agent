# mlflow-edd-financial-agent

A financial AI agent built with LangGraph and local Qwen 3.6 35B (LMStudio), utilizing MCP Yahoo Finance tools. Showcases MLflow Evaluation-Driven Development (EDD) with:

- Persistent SQLite tracing (`sqlite:///mlflow.db`) and LangGraph autolog
- Gemini LLM judges via MLflow `make_judge` (`gemini:/gemini-2.5-pro`) and `mlflow.genai.evaluate`
- Expert judge alignment (MemAlign) with registered judge versions
- Versioned golden evaluation datasets, baseline regression gates, and side-by-side trace comparison before vs after calibration

Project governance: `.specify/memory/constitution.md` (v1.1.0).  
Feature docs: [`specs/001-edd-financial-assistant/quickstart.md`](specs/001-edd-financial-assistant/quickstart.md).

## Prerequisites

- Python >=3.13 (`uv sync`)
- Copy env template and set keys: `cp .env.example .env` (then edit `.env`)
- Cursor/VS Code injects `.env` into new integrated terminals (`python.terminal.useEnvFile`)
- For any other shell: `set -a && source .env && set +a`
- [LMStudio](https://lmstudio.ai) serving `qwen/qwen3.6-35b-a3b` at `http://localhost:1234/v1`
- `GEMINI_API_KEY` in `.env` for judges / MemAlign reflection **and** embeddings
  (`MEMALIGN_EMBEDDING_MODEL=gemini:/text-embedding-004` by default)
- `OPENAI_API_KEY` only if you override the embedding pin to an `openai:/` URI

## Commands

```bash
# Single analysis (P1)
uv run python main.py run-agent --ticker AAPL

# Golden baseline + uncalibrated Gemini judges (P2)
uv run python main.py run-baseline-eval --dataset-version v1

# Review / annotate (primary human path)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Demo seed mimicking UI overrides (optional)
uv run python main.py seed-feedback --file data/expert_feedback_seed.json

# Align selected judge(s) + re-evaluate (P3); baseline retained
uv run python main.py align-and-reeval --judges ToolCallEfficiency --alignment-round 1
```

## Privacy

Only public-market prompts, assistant Markdown, and tool summaries needed for scoring are sent to Gemini. No private account or customer PII.

## Growing the golden suite

When failures or accepted human feedback identify gaps, bump `DATASET_VERSION` (e.g. `v2`), add cases in `golden_dataset.py`, and re-register the dataset. Do not silently overwrite a frozen baseline version still used for comparison.

## Modules

| File | Role |
|------|------|
| `mcp_server.py` | FastMCP + yfinance tools |
| `agent.py` | LangGraph ReAct + LMStudio + autolog |
| `golden_dataset.py` | Versioned MLflow eval dataset |
| `eval_pipeline.py` | Judges, MemAlign, evaluate |
| `main.py` | Orchestration CLI |
| `config.py` | URIs, model pins, version tags |

## Acceptance dry-run (SC-001–SC-006)

1. `run-agent` returns required Markdown sections with tool-backed content
2. `run-baseline-eval` completes 10 cases with durable traces
3. Uncalibrated scores exist for ToolCallEfficiency, ToolCallCorrectness, Groundedness
4. After five overrides + align, aligned eval is tagged distinctly on the same `dataset_version`
5. Full cycle fits one session; second cycle does not erase baseline
6. Aligned judge agrees with ≥4/5 seeded/UI overrides for the selected judge
