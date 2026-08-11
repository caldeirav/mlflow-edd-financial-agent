# Implementation Plan: EDD Financial Assistant

**Branch**: `001-edd-financial-assistant` | **Date**: 2026-08-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-edd-financial-assistant/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a local LangGraph ReAct financial assistant that calls Yahoo Finance tools
via a FastMCP server, returns fixed Markdown analysis sections, and demonstrates
MLflow Evaluation-Driven Development: versioned 10-case golden dataset, Gemini
`gemini-2.5-pro` judges (ToolCallEfficiency, ToolCallCorrectness, Groundedness),
human-override → MemAlign → re-evaluate on the same traces with distinct run tags.

Architecture is five core modules: `mcp_server.py`, `agent.py`,
`golden_dataset.py`, `eval_pipeline.py`, `main.py`, with all metadata in
`sqlite:///mlflow.db`.

## Technical Context

**Language/Version**: Python >=3.13

**Primary Dependencies**: LangGraph, langchain-openai, langchain-mcp-adapters,
mcp/FastMCP, yfinance, mlflow (>=3.15.1), google-genai (provider under MLflow),
dspy/jinja2/tqdm (MemAlign), pytest

**Storage**: MLflow tracking + traces at `sqlite:///mlflow.db`; versioned GenAI
evaluation dataset records in MLflow; no application DB

**Testing**: pytest for tools/dataset helpers; MLflow golden eval as regression
gate (pre/post alignment comparison)

**Target Platform**: Local developer machine (macOS/Linux); LMStudio at
`http://localhost:1234/v1`; Gemini API key for judges

**Project Type**: Single-project CLI/script demo (not a web service)

**Performance Goals**: Full 10-case baseline + judge pass completable in one
operator session; no multi-tenant latency SLO

**Constraints**: Constitution v1.1.0 EDD gates; live market ground truth for
Groundedness; unordered required tool sets; Markdown section checklist;
scripts for agent/eval; MLflow UI primary for human annotate (seeded feedback
dict allowed for reproducible demos — see Complexity Tracking)

**Scale/Scope**: 10 golden cases; 3 qualitative judges; ≥5 human overrides per
alignment round; single local agent model (Qwen 3.6 35B via LMStudio)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Per `.specify/memory/constitution.md` (MLflow EDD Financial Agent v1.1+):

- **I. Persistent Agent Tracing**: PASS — `main.py`/`agent.py` set
  `sqlite:///mlflow.db`, named experiment, `mlflow.langchain.autolog()`.
- **II. Model-as-a-Judge (Gemini)**: PASS — `eval_pipeline.py` uses
  `mlflow.genai.judges.make_judge` with `gemini:/gemini-2.5-pro` and
  `mlflow.genai.evaluate` for ToolCallEfficiency, ToolCallCorrectness,
  Groundedness (+ code scorers for Markdown headings / required tools).
- **III. Judge Alignment (MemAlign)**: PASS — human assessments (UI and/or
  seeded dict → assessments) → `MemAlignOptimizer` with pinned
  `reflection_lm` + `embedding_model` → `align` → `register`.
- **IV. Golden Suite**: PASS — `golden_dataset.py` creates versioned MLflow
  dataset with expectations (required tools, Markdown sections); live tool
  outputs as Groundedness ground truth.
- **V. Regression Gates**: PASS — pre-alignment eval is frozen baseline;
  post-alignment compared on same `dataset_version`.
- **VI. Taxonomy & Roles**: PASS — tags `agent_version`, `judge_version`,
  `dataset_version`, `alignment_round`; Qwen agent vs Gemini judge vs Gemini
  reflection vs pinned embedding.
- **VII. Privacy & Cost Hygiene**: PASS — public Yahoo data + prompts only to
  Gemini; `reflection_lm` uses Gemini Flash-class; no PII/accounts.
- **Stack & workflow**: PASS — five-module layout matches stack constraints.

*Post-design re-check: PASS (see research.md decisions; Complexity Tracking
documents demo feedback-seed path).*

## Project Structure

### Documentation (this feature)

```text
specs/001-edd-financial-assistant/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── mcp-tools.md
│   ├── cli.md
│   └── judges.md
├── checklists/
│   └── requirements.md
└── tasks.md                 # Created by /speckit-tasks (not this command)
```

### Source Code (repository root)

```text
mcp_server.py          # FastMCP + yfinance tools
agent.py               # LangGraph ReAct + LMStudio + autolog + MCP tools
golden_dataset.py      # Versioned MLflow eval dataset + expectations
eval_pipeline.py       # Judges, feedback seed, MemAlign, evaluate pre/post
main.py                # Orchestration runner → sqlite:///mlflow.db
config.py              # URIs, versions, model pins (small shared config)
tests/
├── test_mcp_server.py
├── test_golden_dataset.py
└── test_eval_helpers.py
data/
└── expert_feedback_seed.json   # Optional demo overrides (mimic UI)
pyproject.toml
README.md
mlflow.db                      # Created at runtime (gitignored)
```

**Structure Decision**: Flat five-module layout at repo root per feature
request; thin `config.py` for shared pins; tests for deterministic helpers.
MCP server runs as stdio subprocess from the agent via
`langchain-mcp-adapters`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Seeded expert feedback dictionary in `eval_pipeline.py` (in addition to MLflow UI annotations) | Spec requires MLflow UI as primary annotate path, but demos/CI need reproducible ≥5 overrides without manual UI clicks | UI-only feedback blocks automated end-to-end alignment demos and flake-free tutorials |
| Explicit MemAlign `embedding_model` pin (may be non-Gemini if Gemini embeddings unsupported by MemAlign) | Constitution forbids silent OpenAI default; episodic memory requires an embedding URI | Leaving MemAlign default hides an OpenAI dependency and breaks Gemini-only assumption |
