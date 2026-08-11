# mlflow-edd-financial-agent

Demo of **MLflow Evaluation-Driven Development (EDD)** for a financial research assistant.

The agent uses **LangGraph + local Qwen (LMStudio)** and **MCP Yahoo Finance tools** to write a Markdown analysis. Gemini **LLM judges** score quality; you review traces in the MLflow UI and add expert feedback where you disagree; **MemAlign** calibrates selected judges; a second eval keeps the baseline so you can compare before vs after.

## What the demo shows

1. **Agent** — ReAct loop over market-data tools; Markdown with required sections (Price context, News, Financial statements, Risks/limitations).
2. **Baseline eval** — Versioned golden dataset, durable traces, uncalibrated judges + code scorers.
3. **Human review** — Inspect runs/traces in MLflow UI; override assessments that do not match expert judgment.
4. **Align + re-eval** — MemAlign on the judge(s) you annotated; new aligned run; baseline retained for comparison.

### Scorers

| Scorer | Type | What it checks |
|--------|------|----------------|
| `RequiredMarkdownSections` | Code | Required `##` headings present |
| `RequiredToolsUsed` | Code | Required market tools called |
| `ToolCallCorrectness` | Gemini judge | Right tools for the request |
| `ToolCallEfficiency` | Gemini judge | Lean tool use (no thrash) |
| `Groundedness` | Gemini judge | Claims supported by tool observations |

Judges score **inputs / outputs / expectations** (not raw `{{ trace }}`). Tool evidence is compacted into outputs so qualitative judges can see the same facts as the UI.

## Prerequisites

- Python ≥ 3.13 and [`uv`](https://github.com/astral-sh/uv): `uv sync`
- Copy env: `cp .env.example .env` and set `GEMINI_API_KEY`
- Shell: Cursor/VS Code loads `.env` in integrated terminals; otherwise `set -a && source .env && set +a`
- [LMStudio](https://lmstudio.ai) serving `qwen/qwen3.6-35b-a3b` at `http://localhost:1234/v1`
- MemAlign embeddings default: `MEMALIGN_EMBEDDING_MODEL=gemini:/gemini-embedding-001` (uses `GEMINI_API_KEY`)
- `OPENAI_API_KEY` only if you override the embedding pin to an `openai:/` URI

## Demo walkthrough

The loop is always the same: **run → evaluate → review in UI → align → compare**. Which judge you annotate, and which labels you choose, depends on what you care about in your run.

### 1. Run a single analysis

```bash
uv run python main.py run-agent --ticker AAPL
```

Confirms the agent, tools, and tracing path. Stdio shows tool calls and the Markdown report (use `--quiet` to suppress pretty traces).

### 2. Run a baseline evaluation

```bash
uv run python main.py run-baseline-eval --dataset-version v1
```

Produces an MLflow run (e.g. `baseline-eval-v1`) with one sample per golden case, code-scorer results, and LLM-judge assessments on each trace.

### 3. Review in the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Typical path:

1. Open the experiment **edd-financial-assistant**.
2. Open the **baseline** run.
3. Go to **Traces** (sample-level results live on traces, not only run metrics).
4. Open a trace → inspect agent I/O, tool spans, and existing assessments (value + rationale).
5. Where you disagree with a judge (or want to reinforce a correct call), add a **HUMAN** assessment:
   - Use the **same assessment name** as the judge (e.g. `Groundedness`, `ToolCallEfficiency`).
   - Set the label/value you want the aligned judge to learn.
   - Add a short rationale — MemAlign uses it.

Aim for several HUMAN assessments on the judge(s) you plan to align (often ~5+). You do not need to annotate every scorer or every sample.

Optional: seed file-based feedback instead of (or in addition to) UI clicks:

```bash
uv run python main.py seed-feedback --file data/expert_feedback_seed.json
```

Keep the same `dataset_version` through alignment so baseline and aligned runs stay comparable. Bump the golden suite only when you change cases.

### 4. Align selected judges and re-evaluate

```bash
uv run python main.py align-and-reeval --judges <JudgeName> [--judges <Other>] \
  --dataset-version v1 --alignment-round 1
```

Example: `--judges Groundedness` or `--judges ToolCallEfficiency`.

This MemAligns from HUMAN assessments on traces, then runs a new eval tagged as aligned. The baseline run is **not** deleted.

#### What alignment does under the hood

Alignment does **not** fine-tune Gemini weights. It builds an **episodic memory** of your expert feedback and attaches that memory to the judge so future scores can retrieve similar past cases.

Rough pipeline when you run `align-and-reeval`:

1. **Collect feedback traces** — The CLI finds traces that already have a HUMAN assessment whose name matches the judge (e.g. `Groundedness`). Empty agent outputs are patched with a placeholder so MemAlign can still treat “no report” as a valid example.
2. **Turn traces into training examples** — For each matching assessment, MLflow extracts the request/response (and expectations if the judge uses them), plus your label and rationale.
3. **Embed + store** — `MemAlignOptimizer` embeds those examples (`MEMALIGN_EMBEDDING_MODEL`, default `gemini:/gemini-embedding-001`) into a retriever so similar future cases can be looked up.
4. **Reflect (optional distillation)** — A cheaper reflection LM (`MEMALIGN_REFLECTION_LM`, default Gemini Flash) helps turn feedback into reusable guidance stored with the memory-augmented judge.
5. **Register** — The aligned judge is registered on the experiment so later evals can use the calibrated version.
6. **Re-evaluate** — The golden dataset is scored again with the aligned judge(s) mixed with any unchanged scorers. The new run is tagged `eval_phase=aligned` (and an `alignment_round`); the original baseline stays for comparison.

At **inference/eval time**, the aligned judge retrieves nearby human-labeled examples and uses them as few-shot / memory context when deciding the new label — so it tends to follow your policy (e.g. how you treat empty reports or invents) without rewriting the base judge prompt by hand.

### 5. Compare before vs after

Back in the UI:

1. Keep both **uncalibrated** and **aligned** runs visible (same `dataset_version`).
2. Open **Traces** on each run (or compare the same tickers/cases side by side).
3. Focus on the judge(s) you aligned: did labels/rationales move toward your HUMAN feedback?
4. Spot-check agreement with the assessments you entered (target: most of your overrides still match).

Further rounds: add more HUMAN feedback, then `--alignment-round 2` (etc.). Baselines remain for regression comparison.

### 6. Optional: rescore without re-running the agent

Re-judge existing baseline traces (e.g. after evidence/judge plumbing changes):

```bash
uv run python main.py rescore-eval --run-id <baseline-run-id>
uv run python main.py rescore-eval --run-id <baseline-run-id> --judges Groundedness --judges-only
```

Does not call LMStudio.

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

When review exposes coverage gaps, bump `DATASET_VERSION` (e.g. `v2`), add cases in `golden_dataset.py`, and re-register. Do not silently overwrite a frozen baseline still used for comparison.

## Modules

| File | Role |
|------|------|
| `mcp_server.py` | FastMCP + yfinance tools |
| `agent.py` | LangGraph ReAct + LMStudio + autolog + tool compaction |
| `golden_dataset.py` | Versioned MLflow eval dataset |
| `eval_pipeline.py` | Code scorers, Gemini judges, MemAlign, rescore, evaluate |
| `console_trace.py` | Pretty stdio for agent/judge steps |
| `main.py` | Orchestration CLI |
| `config.py` | URIs, model pins, version tags |

## Acceptance checklist

1. Single-agent run returns required Markdown with tool-backed content  
2. Baseline eval completes the golden set with durable traces  
3. Uncalibrated scores exist for the qualitative judges you care about  
4. After HUMAN overrides + align, an aligned eval is tagged distinctly on the same `dataset_version`  
5. Full cycle fits one session; later rounds do not erase baseline  
6. Aligned judge largely agrees with the HUMAN assessments you provided for that judge  

## Specification artifacts

| Artifact | Path |
|----------|------|
| Project constitution | [`.specify/memory/constitution.md`](.specify/memory/constitution.md) |
| Feature spec | [`specs/001-edd-financial-assistant/spec.md`](specs/001-edd-financial-assistant/spec.md) |
| Implementation plan | [`specs/001-edd-financial-assistant/plan.md`](specs/001-edd-financial-assistant/plan.md) |
| Research notes | [`specs/001-edd-financial-assistant/research.md`](specs/001-edd-financial-assistant/research.md) |
| Data model | [`specs/001-edd-financial-assistant/data-model.md`](specs/001-edd-financial-assistant/data-model.md) |
| Quickstart (spec kit) | [`specs/001-edd-financial-assistant/quickstart.md`](specs/001-edd-financial-assistant/quickstart.md) |
| Tasks | [`specs/001-edd-financial-assistant/tasks.md`](specs/001-edd-financial-assistant/tasks.md) |
| Contracts | [`specs/001-edd-financial-assistant/contracts/`](specs/001-edd-financial-assistant/contracts/) |
