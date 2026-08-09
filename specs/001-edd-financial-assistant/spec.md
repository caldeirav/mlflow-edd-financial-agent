# Feature Specification: EDD Financial Assistant

**Feature Branch**: `001-edd-financial-assistant`

**Created**: 2026-08-09

**Status**: Draft

**Input**: User description: "Build a simple financial assistant using LangGraph and local qwen/qwen3.6-35b-a3b (served via LMStudio at http://localhost:1234/v1). Tooling: FastMCP server wrapping Yahoo Finance (yfinance) for stock prices, news and financial statement analysis. Dataset: 10-case golden dataset based on a single request for financial analysis with expected response structure, facts, and model/tool call sequence expectations. Evaluation & Alignment: baseline traces → uncalibrated Gemini judge (ToolCallEfficiency, ToolCallCorrectness, Groundedness) → 5 expert overrides on misclassified redundant tool calls/reasoning thrash → MemAlign judge.align() → re-evaluate baseline under separate dataset experiment tags; allow looping for more feedback."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Request Financial Analysis (Priority: P1)

An analyst asks the financial assistant a natural-language question about a
public company (for example, summarize recent performance, price context,
news, and key financial-statement signals). The assistant gathers market
data through market-data tools and returns a structured analysis the analyst
can review.

**Why this priority**: Without a working assistant and tool-backed answers,
there is nothing meaningful to evaluate or align.

**Independent Test**: Submit one analysis request for a known ticker and
confirm the response includes tool-backed price, news, and statement-related
content in the expected structure.

**Acceptance Scenarios**:

1. **Given** the assistant and market-data tools are available, **When** the
   analyst submits a financial analysis request for a ticker, **Then** the
   assistant returns a structured analysis that cites tool-derived facts
   (price, news, and/or statements as relevant).
2. **Given** a request that needs multiple data types, **When** the assistant
   runs, **Then** it uses market-data tools rather than inventing figures.
3. **Given** market data is temporarily unavailable for a ticker, **When** the
   assistant cannot complete a tool call, **Then** it reports the limitation
   clearly instead of fabricating numbers.

---

### User Story 2 - Run Golden Baseline Evaluation (Priority: P2)

An evaluation operator runs the fixed 10-case golden analysis suite against
the assistant, captures full execution records for each case, and scores them
with an uncalibrated Gemini qualitative judge on ToolCallEfficiency,
ToolCallCorrectness, and Groundedness. Results are stored so baseline quality
is visible and comparable later.

**Why this priority**: Establishes the measurable baseline required before
judge alignment and before-vs-after comparison.

**Independent Test**: Execute the 10 golden cases once and confirm each case
has a durable execution record plus judge scores for all three metrics,
labeled as the uncalibrated/baseline evaluation.

**Acceptance Scenarios**:

1. **Given** the golden suite of 10 cases is defined, **When** the operator
   runs baseline evaluation, **Then** all 10 cases execute and produce durable
   traces with tool/model call detail.
2. **Given** baseline traces exist, **When** the uncalibrated judge runs,
   **Then** each case receives ToolCallEfficiency, ToolCallCorrectness, and
   Groundedness assessments.
3. **Given** baseline evaluation completes, **When** the operator reviews
   experiment metadata, **Then** the run is tagged as the uncalibrated/baseline
   evaluation for the declared dataset version.

---

### User Story 3 - Align Judge With Expert Feedback and Re-Score (Priority: P3)

An expert reviews baseline traces where the uncalibrated judge mislabeled
redundant tool calls or reasoning thrash, records five human override
annotations with rationale, aligns the judge from that feedback, and
re-evaluates the same baseline traces with the aligned judge. Both evaluation
runs remain distinguishable for side-by-side comparison. The operator may
repeat feedback → align → re-evaluate as needed.

**Why this priority**: Demonstrates Evaluation-Driven Development value:
calibrating judges to expert judgment and proving improvement on the same
cases.

**Independent Test**: Apply five expert overrides on known misclassifications,
run alignment, re-score the same baseline traces, and confirm a separately
tagged aligned evaluation exists alongside the baseline evaluation.

**Acceptance Scenarios**:

1. **Given** baseline judge scores that incorrectly flag redundant tool calls
   or reasoning thrash, **When** an expert adds five human overrides with
   rationale on those traces, **Then** the overrides are stored as human
   assessments tied to the matching judge names.
2. **Given** those human assessments, **When** alignment runs, **Then** an
   aligned judge is produced and registered for reuse, capturing semantic
   rules and episodic exemplars from the feedback.
3. **Given** an aligned judge, **When** the same baseline traces are
   re-evaluated, **Then** scores are logged under a distinct evaluation tag
   from the uncalibrated run (same dataset version) for comparison.
4. **Given** further disagreements remain, **When** the operator collects more
   human feedback and repeats align → re-evaluate, **Then** a new alignment
   round is recorded without overwriting prior baseline evidence.

---

### Edge Cases

- Market-data tool returns empty, delayed, or partial results for a ticker.
- Requested ticker is invalid or delisted.
- Local assistant model endpoint is unreachable.
- Judge service is unavailable or returns malformed assessments.
- Fewer than five clear misclassifications exist in the first baseline pass
  (operator may seed known thrash examples or expand review until five
  overrides are obtained).
- Human override conflicts with an earlier override (latest accepted rationale
  wins for the next alignment round; prior rounds remain auditable).
- Alignment attempted with insufficient or same-sign-only labels (system
  warns and does not claim a successful alignment).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a financial assistant that accepts a
  natural-language financial analysis request and returns a structured
  analysis response.
- **FR-002**: System MUST obtain stock prices, news, and financial-statement
  information through a market-data tool service (Yahoo Finance–backed), not
  by inventing market facts.
- **FR-003**: System MUST use the project’s local assistant model endpoint for
  agent reasoning (Qwen 3.6 35B class model served locally for development).
- **FR-004**: System MUST define a golden evaluation suite of exactly 10 cases
  derived from one canonical financial-analysis request pattern, varying
  tickers/scenarios as needed.
- **FR-005**: Each golden case MUST include: the request, expected response
  structure, expected factual anchors, and expected model/tool-call sequence
  expectations.
- **FR-006**: System MUST run a baseline execution of all golden cases and
  persist full execution traces for each case.
- **FR-007**: System MUST evaluate baseline traces with an uncalibrated Gemini
  `gemini-2.5-pro` qualitative judge producing three named metrics:
  ToolCallEfficiency, ToolCallCorrectness, and Groundedness.
- **FR-008**: System MUST allow experts to attach at least five human override
  annotations (with rationale) on traces where the uncalibrated judge
  misclassified redundant tool calls or reasoning thrash.
- **FR-009**: System MUST align the affected judge(s) from those human
  assessments using MemAlign-style alignment so semantic rules and episodic
  exemplars are retained, then register the aligned judge for reuse.
- **FR-010**: System MUST re-evaluate the same baseline traces with the aligned
  judge and log uncalibrated vs aligned evaluation runs under separate
  dataset/experiment tags for comparison.
- **FR-011**: System MUST support repeating the feedback → align → re-evaluate
  loop for additional alignment rounds without destroying prior baseline
  evidence.
- **FR-012**: When tools or the local model are unavailable, the assistant MUST
  fail gracefully with an actionable error rather than fabricating analysis.

### Evaluation-Driven Development Requirements *(mandatory for agent features)*

Per project constitution:

- **EDD-001**: System MUST persist traces to `sqlite:///mlflow.db` with
  `mlflow.langchain.autolog()` (or equiv.) and typed `AGENT`/`TOOL`/`LLM` spans.
- **EDD-002**: Qualitative eval MUST use `make_judge` with
  `gemini:/gemini-2.5-pro` (or equiv. Gemini URI) via `mlflow.genai.evaluate`,
  with scorers ToolCallEfficiency, ToolCallCorrectness, and Groundedness
  (mapping constitution tool-efficiency, financial-reasoning/correctness, and
  groundedness requirements).
- **EDD-003**: System MUST support MemAlign (`human assessments` → `align` →
  `register`, plus `unalign`) with pinned `reflection_lm` and `embedding_model`.
- **EDD-004**: System MUST maintain a versioned golden evaluation dataset
  (`dataset_version` for the 10-case suite) and persist traces for same-dataset
  UI comparison.
- **EDD-005**: System MUST be able to grow `dataset_version` when failures or
  accepted human feedback identify gaps (initial delivery ships 10 cases;
  growth process is required even if the first bump happens after v1 demo).
- **EDD-006**: Aligned-judge evaluation MUST be compared against the frozen
  uncalibrated baseline on the same dataset; delivery MUST NOT claim success if
  comparison evidence is missing.
- **EDD-007**: Eval runs MUST record `agent_version`, `judge_version`,
  `dataset_version`, and `alignment_round` (baseline round = 0 or equivalent).
- **EDD-008**: Human corrections MUST be stored as MLflow assessments with a
  human source on the relevant traces, using judge names that match the scorers
  being aligned.
- **EDD-009**: Only public-market tool outputs and assistant/judge prompts
  needed for scoring may be sent to Gemini; no private account or customer PII
  is in scope for this feature.

### Key Entities

- **Analysis Request**: Natural-language financial analysis prompt and ticker
  or subject company context.
- **Assistant Response**: Structured analysis output intended to match the
  golden expected response shape.
- **Market Data Observation**: Tool-returned prices, news items, and statement
  figures referenced by the assistant.
- **Golden Case**: One suite item with request, expected structure, factual
  anchors, and expected tool/model call sequence.
- **Execution Trace**: Durable record of an assistant run including model and
  tool calls.
- **Judge Assessment**: Qualitative score and rationale for a named metric on a
  trace (uncalibrated or aligned).
- **Human Override**: Expert assessment correcting a judge misclassification,
  with rationale.
- **Alignment Round**: Bundled human feedback, aligned judge version, and
  re-evaluation run metadata.
- **Evaluation Run Tags**: Distinguishing labels for uncalibrated vs aligned
  scoring of the same dataset version.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An analyst can obtain a structured, tool-backed financial
  analysis for a supported public ticker in a single request without manual
  data lookup.
- **SC-002**: 100% of the 10 golden cases complete baseline execution with
  durable traces suitable for later comparison.
- **SC-003**: 100% of baseline traces receive all three qualitative scores
  (efficiency, correctness, groundedness) in the uncalibrated evaluation.
- **SC-004**: After five expert overrides and alignment, the aligned evaluation
  is available side-by-side with the uncalibrated evaluation on the same 10
  cases, with clear before/after labeling.
- **SC-005**: Operators can complete at least one full feedback → align →
  re-evaluate cycle in a single working session, and can start a second cycle
  without losing the original baseline results.
- **SC-006**: In a review of the five overridden cases, aligned judge outcomes
  agree with the expert overrides on at least 4 of 5 cases after the first
  alignment round.

### Evaluation Outcomes *(mandatory for agent features)*

- **SC-EDD-001**: Every golden-case run under test produces durable typed
  traces in the local tracking store, viewable in the local experiment UI.
- **SC-EDD-002**: Qualitative scores come from named Gemini `gemini-2.5-pro`
  judges through the project’s standard evaluate/scorer path (no alternate
  qualitative judge path).
- **SC-EDD-003**: After the five human overrides, MemAlign align + register
  produces a new `judge_version`; re-evaluation uses that version and records
  `alignment_round`.
- **SC-EDD-004**: Golden eval uses a declared `dataset_version` for the 10-case
  suite; uncalibrated and aligned runs share that dataset version for
  comparison.
- **SC-EDD-005**: Process exists to bump `dataset_version` from failures or
  accepted human feedback before closing related quality issues.
- **SC-EDD-006**: Delivery evidence includes uncalibrated baseline vs aligned
  comparison on the same dataset (regression/comparison gate for this demo).
- **SC-EDD-007**: Eval runs are attributable via `agent_version`,
  `judge_version`, `dataset_version`, and `alignment_round` tags.

## Assumptions

- Local LMStudio serves `qwen/qwen3.6-35b-a3b` (or equivalent configured id) at
  `http://localhost:1234/v1` during development and demo runs.
- Market data comes from a FastMCP server wrapping Yahoo Finance (`yfinance`)
  for prices, news, and financial statements.
- Orchestration uses LangGraph per project constitution; tracing uses the
  project SQLite MLflow store.
- The “single request” pattern is one canonical analysis prompt template;
  the 10 cases vary company/ticker and lightly vary focus while sharing the
  same expected response structure schema.
- ToolCallCorrectness covers appropriate tool choice and argument use;
  ToolCallEfficiency covers redundant calls and reasoning thrash;
  Groundedness covers faithfulness to tool-returned facts (fulfills
  constitution groundedness / numerical consistency intent).
- Initial alignment uses five expert overrides focused on efficiency/thrash
  misclassifications; additional rounds may add labels on other metrics.
- MemAlign `reflection_lm` will be pinned to a Gemini-family model;
  `embedding_model` will be explicitly pinned at implementation (no silent
  OpenAI default) per constitution.
- This feature is a local demo/showcase: no multi-user auth, no brokerage
  trading, and no private customer portfolio data.
- “Separate dataset experiment tags” means distinct evaluation-run tags /
  metadata separating uncalibrated vs aligned scoring while sharing the same
  `dataset_version`.
