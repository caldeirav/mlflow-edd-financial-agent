---
description: "Task list for EDD financial assistant implementation"
---

# Tasks: EDD Financial Assistant

**Input**: Design documents from `/specs/001-edd-financial-assistant/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Optional — not explicitly required by the feature spec; polish may add helper tests later.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: modules at repository root (`mcp_server.py`, `agent.py`, …) per plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and shared config

- [ ] T001 Add runtime ignores for `mlflow.db`, `.mlflow/`, and local env files in `.gitignore`
- [ ] T002 Add dependencies `langchain-mcp-adapters`, `dspy`, `jinja2`, `tqdm` (and `fastmcp` if needed) in `pyproject.toml` and refresh the lockfile
- [ ] T003 [P] Create shared pins and version tags defaults in `config.py` (tracking URI `sqlite:///mlflow.db`, experiment name, LMStudio URL/model, Gemini judge URI, MemAlign `reflection_lm` + explicit `embedding_model`, `agent_version`/`dataset_version`)
- [ ] T004 [P] Update `README.md` with quickstart pointers to `specs/001-edd-financial-assistant/quickstart.md` and required env vars (`GEMINI_API_KEY`, LMStudio)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: MLflow/EDD scaffolding shared by all stories

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Implement MLflow init helper in `config.py` (or small `mlflow_setup` section): `set_tracking_uri`, `set_experiment`, and helpers to set run tags `agent_version`, `judge_version`, `dataset_version`, `alignment_round`, `eval_phase`
- [ ] T006 [P] Document Gemini-bound payload policy (public market prompts/outputs/tool summaries only) in `README.md` and comments in `eval_pipeline.py` stub header
- [ ] T007 Create empty module stubs `mcp_server.py`, `agent.py`, `golden_dataset.py`, `eval_pipeline.py` and replace Hello-world `main.py` with argparse skeleton for `run-agent`, `run-baseline-eval`, `seed-feedback`, `align-and-reeval` per `specs/001-edd-financial-assistant/contracts/cli.md`

**Checkpoint**: Foundation ready — user story implementation can begin

---

## Phase 3: User Story 1 - Request Financial Analysis (Priority: P1) 🎯 MVP

**Goal**: Analyst can request a tool-backed Markdown financial analysis via local LangGraph agent + FastMCP Yahoo tools

**Independent Test**: `python main.py run-agent --ticker AAPL` returns Markdown with Price context, News, Financial statements, Risks/limitations and shows tool usage in an MLflow trace

### Implementation for User Story 1

- [ ] T008 [P] [US1] Implement FastMCP tools `get_stock_price`, `get_financial_news`, and `get_financial_statements` with yfinance in `mcp_server.py` per `specs/001-edd-financial-assistant/contracts/mcp-tools.md`
- [ ] T009 [P] [US1] Add stdio entrypoint so `python mcp_server.py` serves the MCP server in `mcp_server.py`
- [ ] T010 [US1] Implement LangGraph ReAct agent with LMStudio `ChatOpenAI` client, system prompt enforcing required Markdown headings, and `mlflow.langchain.autolog()` in `agent.py`
- [ ] T011 [US1] Wire MCP tools into the agent via `langchain-mcp-adapters` `MultiServerMCPClient` (stdio → `mcp_server.py`) in `agent.py`
- [ ] T012 [US1] Implement graceful failure when LMStudio or tools are unavailable (no fabricated numbers) in `agent.py`
- [ ] T013 [US1] Implement `run-agent` command in `main.py` calling `agent.py` and initializing MLflow from `config.py`

**Checkpoint**: User Story 1 independently demoable

---

## Phase 4: User Story 2 - Run Golden Baseline Evaluation (Priority: P2)

**Goal**: Operator runs 10-case golden suite with durable traces and uncalibrated Gemini scores for ToolCallEfficiency, ToolCallCorrectness, and Groundedness

**Independent Test**: `python main.py run-baseline-eval --dataset-version v1` completes 10 cases with traces + three judge assessments tagged `eval_phase=uncalibrated`, `alignment_round=0`

### Implementation for User Story 2

- [ ] T014 [P] [US2] Define canonical analysis prompt template and 10 golden case records (tickers/focus variants) with `required_sections` and unordered `required_tools` in `golden_dataset.py`
- [ ] T015 [US2] Create/upsert versioned MLflow GenAI evaluation dataset (`financial_analysis_golden`, `dataset_version=v1`) with inputs/expectations and live `groundedness_policy` in `golden_dataset.py`
- [ ] T016 [P] [US2] Implement code scorers `RequiredMarkdownSections` and `RequiredToolsUsed` in `eval_pipeline.py`
- [ ] T017 [P] [US2] Implement uncalibrated `make_judge` scorers `ToolCallEfficiency`, `ToolCallCorrectness`, and `Groundedness` with `gemini:/gemini-2.5-pro` in `eval_pipeline.py` per `specs/001-edd-financial-assistant/contracts/judges.md`
- [ ] T018 [US2] Implement baseline `mlflow.genai.evaluate` path (predict_fn → agent) with run tags for uncalibrated phase in `eval_pipeline.py`
- [ ] T019 [US2] Implement `run-baseline-eval` in `main.py` wiring `golden_dataset.py` + `eval_pipeline.py`

**Checkpoint**: User Stories 1 and 2 work; baseline evidence exists in MLflow UI

---

## Phase 5: User Story 3 - Align Judge With Expert Feedback and Re-Score (Priority: P3)

**Goal**: Operator/expert applies ≥5 overrides (UI or seed), MemAlign-aligns selected judge(s), re-evaluates same dataset with distinct tags

**Independent Test**: After five overrides, `python main.py align-and-reeval --judges ToolCallEfficiency --alignment-round 1` registers aligned judge and logs `eval_phase=aligned` scores without deleting uncalibrated baseline

### Implementation for User Story 3

- [ ] T020 [P] [US3] Create demo seed file `data/expert_feedback_seed.json` with ≥5 override records (trace selectors, judge name, value, rationale)
- [ ] T021 [US3] Implement `seed-feedback` to attach `AssessmentSourceType.HUMAN` assessments onto baseline traces from the seed file in `eval_pipeline.py` and `main.py`
- [ ] T022 [US3] Implement MemAlign path: pin `reflection_lm` + `embedding_model`, `judge.align(...)`, `register(...)`, optional `unalign`, operator-selected `--judges` list in `eval_pipeline.py`
- [ ] T023 [US3] Implement aligned re-evaluation/rescoring on same `dataset_version` with tags `eval_phase=aligned`, bumped `judge_version`/`alignment_round` in `eval_pipeline.py`
- [ ] T024 [US3] Implement `align-and-reeval` command in `main.py` and document MLflow UI annotate + script align flow in `README.md`
- [ ] T025 [US3] Add helper to compare uncalibrated vs aligned score summaries (same dataset) and print/fail soft gate messaging in `eval_pipeline.py` or `main.py`

**Checkpoint**: Full EDD loop demoable; baseline retained alongside aligned run

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Hardening and delivery evidence

- [ ] T026 [P] Ensure `.gitignore` covers `mlflow.db` and document `mlflow ui --backend-store-uri sqlite:///mlflow.db` in `README.md`
- [ ] T027 Align `specs/001-edd-financial-assistant/quickstart.md` commands with final `main.py` CLI flags
- [ ] T028 Verify end-to-end dry-run checklist against SC-001–SC-006 in `README.md` (or `specs/001-edd-financial-assistant/quickstart.md`)
- [ ] T029 [P] Add process note for bumping `dataset_version` when growing the suite in `golden_dataset.py` docstring / `README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup — **BLOCKS** all user stories
- **US1 (Phase 3)**: Depends on Foundational — MVP
- **US2 (Phase 4)**: Depends on Foundational; practically needs US1 agent/tools for `predict_fn`
- **US3 (Phase 5)**: Depends on US2 baseline traces + judges
- **Polish (Phase 6)**: After desired stories complete

### User Story Dependencies

- **US1**: Independent after foundation (MCP + agent only)
- **US2**: Uses US1 agent as evaluate `predict_fn`
- **US3**: Uses US2 baseline traces and judge definitions

### Parallel Opportunities

- T003/T004 in Setup
- T008/T009 vs later agent work once stubs exist; T008∥T009
- T014∥T016∥T017 after US1 complete
- T020 can start once baseline trace id conventions are known

### Parallel Example: User Story 2

```bash
# After US1 works:
Task: "Define 10 golden cases in golden_dataset.py"
Task: "Implement code scorers in eval_pipeline.py"
Task: "Implement three make_judge scorers in eval_pipeline.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1–2
2. Complete Phase 3 (US1)
3. **STOP and VALIDATE**: `run-agent --ticker AAPL` + MLflow trace

### Incremental Delivery

1. US1 → live assistant demo
2. US2 → golden baseline + uncalibrated judges
3. US3 → MemAlign + before/after comparison
4. Polish → docs and suite-growth notes

### Notes

- Prefer `mlflow.genai.judges.make_judge` import path (see research.md)
- Tool name vocabulary must match golden `required_tools` and MCP contract
- Do not delete uncalibrated baseline evidence when aligning
