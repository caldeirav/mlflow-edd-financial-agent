# mlflow-edd-financial-agent

Demo of **MLflow Evaluation-Driven Development (EDD)** for a financial research assistant.

The agent uses **LangGraph + local Qwen (LMStudio)** and **MCP Yahoo Finance tools** to write a Markdown analysis. Gemini **LLM judges** score tool use and groundedness; you override a handful of assessments in the MLflow UI; **MemAlign** calibrates the judge; a second eval keeps the baseline so you can compare before vs after.

Governance: [`.specify/memory/constitution.md`](.specify/memory/constitution.md) (v1.1.0).  
Design artifacts: [`specs/001-edd-financial-assistant/`](specs/001-edd-financial-assistant/).

## What the demo shows

1. **Agent** — ReAct loop over `get_stock_price`, `get_financial_news`, `get_financial_statements`; required Markdown sections: Price context, News, Financial statements, Risks/limitations.
2. **Baseline eval** — 10 golden tickers (`v1`), durable traces in `sqlite:///mlflow.db`, uncalibrated Gemini judges + code scorers.
3. **Human review** — Override judge labels in the UI (especially Groundedness).
4. **Align + re-eval** — MemAlign on selected judges; new `aligned-eval-*` run; baseline retained for side-by-side comparison.

### Scorers

| Scorer | Type | What it checks |
|--------|------|----------------|
| `RequiredMarkdownSections` | Code | Required `##` headings present |
| `RequiredToolsUsed` | Code | All three market tools called |
| `ToolCallCorrectness` | Gemini judge | Right tools for the request |
| `ToolCallEfficiency` | Gemini judge | Lean tool use (no thrash) |
| `Groundedness` | Gemini judge | Claims supported by tool observations |

Judges score **inputs / outputs / expectations** (not raw `{{ trace }}`) so Gemini can return structured feedback. Tool evidence is **compacted** into outputs (news titles/summaries, key statement rows) so Groundedness does not miss late articles from blind truncation.

## Prerequisites

- Python ≥ 3.13 and [`uv`](https://github.com/astral-sh/uv): `uv sync`
- Copy env: `cp .env.example .env` and set `GEMINI_API_KEY`
- Shell: Cursor/VS Code loads `.env` in integrated terminals; otherwise `set -a && source .env && set +a`
- [LMStudio](https://lmstudio.ai) serving `qwen/qwen3.6-35b-a3b` at `http://localhost:1234/v1`
- MemAlign embeddings default: `MEMALIGN_EMBEDDING_MODEL=gemini:/gemini-embedding-001` (uses `GEMINI_API_KEY`)
- `OPENAI_API_KEY` only if you override the embedding pin to an `openai:/` URI

## Demo walkthrough

### 1. Smoke the agent

```bash
uv run python main.py run-agent --ticker AAPL
```

**Check:** Markdown has all four sections; facts come from tools (pretty stdio shows tool calls unless `--quiet`).

### 2. Baseline golden eval

```bash
uv run python main.py run-baseline-eval --dataset-version v1
```

**Check:** Run name like `baseline-eval-v1`; 10 samples scored; traces in MLflow. Expect occasional empty reports (agent variance) and occasional true Groundedness fails (e.g. inventing “surged” when numbers fell).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the baseline run → **Traces** → inspect assessments and rationales.

### 3. Annotate (≥5 HUMAN assessments)

Prefer a **Groundedness** round. On each sample, add a HUMAN assessment named exactly `Groundedness`:

| Situation | Label | Why |
|-----------|-------|-----|
| Empty / missing report | `ungrounded` | Vacuous “grounded” is wrong for this task |
| Real invent (numbers/news not in tools) | `ungrounded` | Affirm the judge if it already caught it |
| Tool-faithful report | `grounded` | Teach the positive class |

Keep `dataset_version=v1` for alignment. Only bump the golden suite when you add/change cases.

Optional demo seed (ToolCallEfficiency-oriented):

```bash
uv run python main.py seed-feedback --file data/expert_feedback_seed.json
```

### 4. Align + re-evaluate

```bash
uv run python main.py align-and-reeval --judges Groundedness --dataset-version v1 --alignment-round 1
```

**Check:**

- New run `aligned-eval-v1-r1` (or similar); baseline still present
- CLI prints both uncalibrated and aligned evidence for the same `dataset_version`
- Side-by-side in UI: same cases, compare Groundedness before vs after
- Aligned judge agrees with ≥4/5 of your HUMAN overrides
- Empty-report policy and invent detection should improve or stay correct

Second round (more annotations): bump `--alignment-round 2` — does not erase baseline.

### 5. Optional: rescore without re-running the agent

Rebuild compacted tool observations from an existing baseline run and re-judge:

```bash
uv run python main.py rescore-eval --run-id <baseline-run-id>
uv run python main.py rescore-eval --run-id <baseline-run-id> --judges Groundedness --judges-only
```

Useful after judge/evidence fixes; does not invoke LMStudio.

## CLI reference

```bash
uv run python main.py run-agent --ticker AAPL
uv run python main.py run-baseline-eval --dataset-version v1
uv run python main.py seed-feedback --file data/expert_feedback_seed.json
uv run python main.py rescore-eval --run-id <run-id>
uv run python main.py align-and-reeval --judges Groundedness --dataset-version v1 --alignment-round 1

# Global: --quiet  (or EDD_QUIET=1) disables pretty stdio traces
```

## Privacy

Only public-market prompts, assistant Markdown, and tool summaries needed for scoring go to Gemini. No private account or customer PII.

## Growing the golden suite

When gaps show up in review, bump `DATASET_VERSION` (e.g. `v2`), add cases in `golden_dataset.py`, and re-register. Do not silently overwrite a frozen baseline still used for comparison.

## Modules

| File | Role |
|------|------|
| `mcp_server.py` | FastMCP + yfinance tools |
| `agent.py` | LangGraph ReAct + LMStudio + autolog + tool compaction |
| `golden_dataset.py` | Versioned MLflow eval dataset (10 tickers in `v1`) |
| `eval_pipeline.py` | Code scorers, Gemini judges, MemAlign, rescore, evaluate |
| `console_trace.py` | Pretty stdio for agent/judge steps |
| `main.py` | Orchestration CLI |
| `config.py` | URIs, model pins, version tags |

## Acceptance checklist

1. `run-agent` returns required Markdown with tool-backed content  
2. `run-baseline-eval` completes 10 cases with durable traces  
3. Uncalibrated scores exist for ToolCallEfficiency, ToolCallCorrectness, Groundedness  
4. After ≥5 overrides + align, aligned eval is tagged distinctly on the same `dataset_version`  
5. Full cycle fits one session; second cycle does not erase baseline  
6. Aligned judge agrees with ≥4/5 UI/seed overrides for the selected judge  
