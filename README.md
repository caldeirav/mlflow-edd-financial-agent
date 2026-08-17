# mlflow-edd-financial-agent

Demo of **MLflow Evaluation-Driven Development (EDD)** for a financial research assistant — aimed at two problems that show up whenever **MCP** moves past toy demos in enterprise settings:

1. **Instrumenting tool calling** so you can see what was called, with which arguments, and what came back (OpenTelemetry-style spans via MLflow / `mlflow.langchain.autolog()`, against a real MCP server — not a mock tool list).
2. **Evaluating and optimizing tool use with continuous learning from human feedback (CLHF)** — tool **selection / calling accuracy**, and **quality of tool-backed answers** (groundedness against live tool observations).

The vehicle is a LangGraph agent on local Qwen (LMStudio) with **MCP Yahoo Finance tools** (`mcp_server.py` over stdio). Gemini judges score each baseline; you review traces in the MLflow UI and add expert assessments where you disagree; **MemAlign** calibrates selected judges; a second eval keeps the baseline so you can compare before vs after.

## Demo

The recording walks through **one agent run**, then the MLflow evaluation of **two runs on the same baseline dataset**: uncalibrated Groundedness, then the same suite after human feedback and alignment.

<p align="center">
  <a href="https://www.youtube.com/watch?v=5KEfejbdQAc">
    <img src="https://img.youtube.com/vi/5KEfejbdQAc/hqdefault.jpg" alt="Watch the EDD financial agent demo on YouTube" width="720">
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=5KEfejbdQAc"><strong>Watch on YouTube</strong></a> — Evaluation-Driven Development with MLflow: agent run, then before vs after Groundedness alignment
</p>

## What the demo shows

1. **MCP + OTel instrumentation** — FastMCP server (`mcp_server.py`) exposing market tools; LangGraph binds them via `langchain-mcp-adapters`; `mlflow.langchain.autolog()` records AGENT / TOOL / LLM spans (args and results) into `sqlite:///mlflow.db` for UI inspection.
2. **Agent behavior** — ReAct loop over those tools; Markdown with required sections (Price context, News, Financial statements, Risks/limitations).
3. **Baseline eval of tool calling & answer quality** — Versioned golden dataset; code scorers plus uncalibrated Gemini judges for selection/efficiency and groundedness.
4. **CLHF in the MLflow UI** — Inspect runs/traces; add HUMAN assessments where judges miss (or correctly catch) tool-use or grounding issues.
5. **Align + re-eval** — MemAlign on the judge(s) you annotated; new aligned run; baseline retained for comparison.

### Scorers

| Scorer | Type | What it checks |
|--------|------|----------------|
| `RequiredMarkdownSections` | Code | Required `##` headings present |
| `RequiredToolsUsed` | Code | Required market tools called (selection coverage) |
| `ToolCallCorrectness` | Gemini judge | Right tools for the request (selection accuracy) |
| `ToolCallEfficiency` | Gemini judge | Lean tool use — no redundant thrash |
| `Groundedness` | Gemini judge | Claims supported by tool observations (execution → answer quality) |

Together, the tool scorers target **calling accuracy**; Groundedness targets whether tool **execution results** were used faithfully in the report. Judges score **inputs / outputs / expectations** (not raw `{{ trace }}`). Tool evidence is compacted into outputs so qualitative judges can see the same facts as the UI.

## Prerequisites

- Python ≥ 3.13 and [`uv`](https://github.com/astral-sh/uv): `uv sync`
- Copy env: `cp .env.example .env` and set `GEMINI_API_KEY`
- Shell: Cursor/VS Code loads `.env` in integrated terminals; otherwise `set -a && source .env && set +a`
- [LMStudio](https://lmstudio.ai) serving `qwen/qwen3.6-35b-a3b` at `http://localhost:1234/v1`
- MemAlign embeddings default: `MEMALIGN_EMBEDDING_MODEL=gemini:/gemini-embedding-001` (uses `GEMINI_API_KEY`)
- `OPENAI_API_KEY` only if you override the embedding pin to an `openai:/` URI

## Demo walkthrough

The loop is always the same: **instrumented run → evaluate tool use & groundedness → CLHF in UI → align → compare**. Which judge you annotate depends on whether you are tuning **tool selection** or **answer quality from tool results** (or both).

### 1. Run a single analysis

```bash
uv run python main.py run-agent --ticker AAPL
```

Starts the MCP stdio server, runs the ReAct agent, and logs OTel-style spans via autolog. Stdio shows tool calls and the Markdown report (use `--quiet` to suppress pretty traces). Implementation reference: `mcp_server.py` (tools) + `agent.py` (`MultiServerMCPClient`, `mlflow.langchain.autolog()`).

### 2. Run a baseline evaluation

```bash
uv run python main.py run-baseline-eval --dataset-version v1
```

Produces an MLflow run (e.g. `baseline-eval-v1`) with one sample per golden case: durable tool-call traces plus scores for selection/efficiency and groundedness.

### 3. Review in the MLflow UI (CLHF)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Typical path:

1. Open the experiment **edd-financial-assistant**.
2. Open the **baseline** run.
3. Go to **Traces** (sample-level tool spans and assessments live on traces, not only run metrics).
4. Open a trace → inspect AGENT / TOOL / LLM spans (MCP tool names, args, observations) and existing assessments (value + rationale).
5. Where you disagree with a judge (or want to reinforce a correct call), add a **HUMAN** assessment — this is the CLHF signal:
   - Use the **same assessment name** as the judge (e.g. `ToolCallCorrectness`, `ToolCallEfficiency`, `Groundedness`).
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
| `mcp_server.py` | FastMCP + yfinance tools (stdio MCP server — instrumentation example) |
| `agent.py` | LangGraph ReAct + LMStudio + MCP client + `mlflow.langchain.autolog()` + tool compaction |
| `golden_dataset.py` | Versioned MLflow eval dataset |
| `eval_pipeline.py` | Tool/groundedness scorers, Gemini judges, MemAlign (CLHF), rescore, evaluate |
| `console_trace.py` | Pretty stdio for agent/judge steps |
| `main.py` | Orchestration CLI |
| `config.py` | URIs, model pins, version tags |

## Acceptance checklist

1. Single-agent run uses the MCP server and returns required Markdown with tool-backed content; traces show TOOL spans  
2. Baseline eval completes the golden set with durable traces and tool/groundedness scores  
3. Uncalibrated scores exist for the qualitative judges you care about (selection and/or groundedness)  
4. After HUMAN overrides (CLHF) + align, an aligned eval is tagged distinctly on the same `dataset_version`  
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
