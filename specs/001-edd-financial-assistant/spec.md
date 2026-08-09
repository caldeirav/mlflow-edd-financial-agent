# Feature Specification: EDD Financial Assistant

**Feature Branch**: `001-edd-financial-assistant`

**Created**: 2026-08-09

**Status**: Draft

**Input**: User description: "Build a simple financial assistant using LangGraph and local qwen/qwen3.6-35b-a3b (served via LMStudio at http://localhost:1234/v1). Tooling: FastMCP server wrapping Yahoo Finance (yfinance) for stock prices, news and financial statement analysis. Dataset: 10-case golden dataset based on a single request for financial analysis with expected response structure, facts, and model/tool call sequence expectations. Evaluation & Alignment: baseline traces → uncalibrated Gemini judge (ToolCallEfficiency, ToolCallCorrectness, Groundedness) → 5 expert overrides on misclassified redundant tool calls/reasoning thrash → MemAlign judge.align() → re-evaluate baseline under separate dataset experiment tags; allow looping for more feedback."

## Clarifications

### Session 2026-08-09

- Q: How should golden-case expected facts relate to live market data? → A: Live validation — at eval time, re-fetch market data and treat current tool output as ground truth
- Q: How strict should golden-case expected tool/model call sequences be for pass/fail? → A: Required tool set only (order flexible; missing required tool fails)
- Q: Which judge(s) should the first MemAlign round calibrate? → A: Operator chooses per round; no fixed default
- Q: What is the primary operator interface for assistant, eval, overrides, and align? → A: Scripts for agent + eval; MLflow UI for review/annotate
- Q: What shape must the assistant’s structured analysis response follow? → A: Fixed Markdown section headings (content free text under each)

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
confirm the response includes the required Markdown sections with tool-backed
price, news, and statement-related content, and that required tools for that
request were invoked (order may vary).

**Acceptance Scenarios**:

1. **Given** the assistant and market-data tools are available, **When** the
   analyst submits a financial analysis request for a ticker, **Then** the
   assistant returns Markdown with the required section headings and cites
   tool-derived facts (price, news, and/or statements as relevant).
2. **Given** a request that needs multiple data types, **When** the assistant
   runs, **Then** it invokes the required market-data tools for that request
   (call order may vary) rather than inventing figures.
3. **Given** market data is temporarily unavailable for a ticker, **When** the
   assistant cannot complete a tool call, **Then** it reports the limitation
   clearly instead of fabricating numbers.

---

### User Story 2 - Run Golden Baseline Evaluation (Priority: P2)

An evaluation operator runs the fixed 10-case golden analysis suite via
scripts against the assistant, captures full execution records for each case,
and scores them with an uncalibrated Gemini qualitative judge on
ToolCallEfficiency, ToolCallCorrectness, and Groundedness. Results are stored
so baseline quality is visible and comparable later in the MLflow UI.

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

An expert reviews baseline traces in the MLflow UI where uncalibrated judges
mislabeled behavior (for example redundant tool calls or reasoning thrash),
records five human override annotations with rationale targeting the judge(s)
the operator selected for this round, then runs alignment (via script once
annotations exist) and re-evaluates the same baseline traces with the aligned
judge(s). Both evaluation runs remain distinguishable for side-by-side
comparison. The operator may repeat feedback → align → re-evaluate and may
choose different judge(s) each round.

**Why this priority**: Demonstrates Evaluation-Driven Development value:
calibrating judges to expert judgment and proving improvement on the same
cases.

**Independent Test**: Apply five expert overrides on known misclassifications
for the operator-selected judge(s), run alignment for those judge(s),
re-score the same baseline traces, and confirm a separately tagged aligned
evaluation exists alongside the baseline evaluation.

**Acceptance Scenarios**:

1. **Given** baseline judge scores the operator disputes, **When** an expert
   adds five human overrides with rationale naming the target judge(s) for
   this round, **Then** the overrides are stored as human assessments tied to
   those judge names.
2. **Given** those human assessments and an operator-selected judge set,
   **When** alignment runs, **Then** each selected judge is aligned and
   registered for reuse, capturing semantic rules and episodic exemplars.
3. **Given** an aligned judge, **When** the same baseline traces are
   re-evaluated, **Then** scores are logged under a distinct evaluation tag
   from the uncalibrated run (same dataset version) for comparison.
4. **Given** further disagreements remain, **When** the operator collects more
   human feedback and repeats align → re-evaluate, **Then** a new alignment
   round is recorded without overwriting prior baseline evidence.

---

### Edge Cases

- Market-data tool returns empty, delayed, or partial results for a ticker
  (Groundedness has no reliable live ground truth until tools succeed).
- Live market values change between agent run and judge scoring (Groundedness
  MUST primarily compare the assistant’s claims to tool outputs on that same
  execution trace; optional live re-fetch must be documented on the eval run
  if used as supplemental context).
- Requested ticker is invalid or delisted.
- Local assistant model endpoint is unreachable.
- Judge service is unavailable or returns malformed assessments.
- Fewer than five clear misclassifications exist in the first baseline pass
  (operator may seed known thrash examples or expand review until five
  overrides are obtained).
- Human override conflicts with an earlier override (latest accepted rationale
  wins for the next alignment round; prior rounds remain auditable).
- Assistant omits a required Markdown section heading (structure checklist
  failure) even if content appears elsewhere in free prose.
- Assistant omits a required tool for a golden case (hard suite failure) even
  if the narrative answer looks plausible.
- Alignment attempted with insufficient or same-sign-only labels (system
  warns and does not claim a successful alignment).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a financial assistant that accepts a
  natural-language financial analysis request and returns a structured
  analysis as Markdown with a fixed set of section headings (content under
  each heading may be free text). Required sections MUST include at least:
  Price context, News, Financial statements, and Risks/limitations.
- **FR-002**: System MUST obtain stock prices, news, and financial-statement
  information through a market-data tool service (Yahoo Finance–backed), not
  by inventing market facts.
- **FR-003**: System MUST use the project’s local assistant model endpoint for
  agent reasoning (Qwen 3.6 35B class model served locally for development).
- **FR-004**: System MUST define a golden evaluation suite of exactly 10 cases
  derived from one canonical financial-analysis request pattern, varying
  tickers/scenarios as needed.
- **FR-005**: Each golden case MUST include: the request, expected Markdown
  section checklist (required headings present), a required tool set
  (unordered; missing a required tool fails), and any structural/factual
  check templates. Call order is not asserted. Expected market facts for
  Groundedness MUST be derived at evaluation time from live tool outputs
  (re-fetched market data treated as ground truth), not from frozen
  historical snapshots.
- **FR-006**: System MUST run a baseline execution of all golden cases and
  persist full execution traces for each case.
- **FR-007**: System MUST evaluate baseline traces with an uncalibrated Gemini
  `gemini-2.5-pro` qualitative judge producing three named metrics:
  ToolCallEfficiency, ToolCallCorrectness, and Groundedness.
- **FR-008**: System MUST allow experts to attach at least five human override
  annotations (with rationale) on traces where the uncalibrated judge
  misclassified behavior (initially expected to include redundant tool calls
  or reasoning thrash). Each override MUST name the target judge/scorer.
- **FR-009**: System MUST align the operator-selected judge(s) for that
  alignment round from matching human assessments using MemAlign-style
  alignment so semantic rules and episodic exemplars are retained, then
  register the aligned judge(s) for reuse. There is no fixed default set of
  judges per round—the operator chooses which of ToolCallEfficiency,
  ToolCallCorrectness, and/or Groundedness to calibrate.
- **FR-010**: System MUST re-evaluate the same baseline traces with the aligned
  judge and log uncalibrated vs aligned evaluation runs under separate
  dataset/experiment tags for comparison.
- **FR-011**: System MUST support repeating the feedback → align → re-evaluate
  loop for additional alignment rounds without destroying prior baseline
  evidence.
- **FR-012**: When tools or the local model are unavailable, the assistant MUST
  fail gracefully with an actionable error rather than fabricating analysis.
- **FR-013**: System MUST provide scripted entrypoints to run the assistant
  (single request and golden-suite execution) and to run uncalibrated/aligned
  evaluations that persist traces and scores. Expert review and human override
  annotation MUST be performed in the MLflow UI (not a separate custom annotate
  CLI as the primary path). Alignment MAY be triggered by a script after UI
  annotations exist.

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
- **Assistant Response**: Markdown analysis with required section headings
  (Price context, News, Financial statements, Risks/limitations); body text
  under each heading is free-form.
- **Market Data Observation**: Live tool-returned prices, news items, and
  statement figures; used as ground truth for Groundedness at evaluation time.
- **Golden Case**: One suite item with request, Markdown section checklist,
  required tool set (unordered), and check templates; market-fact ground truth
  is resolved live from tool outputs at evaluation time.
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

- **SC-001**: An analyst can obtain a Markdown financial analysis with the
  required sections, tool-backed for a supported public ticker, in a single
  request without manual data lookup.
- **SC-002**: 100% of the 10 golden cases complete baseline execution with
  durable traces suitable for later comparison.
- **SC-003**: 100% of baseline traces receive all three qualitative scores
  (efficiency, correctness, groundedness) in the uncalibrated evaluation.
- **SC-004**: After five expert overrides and alignment, the aligned evaluation
  is available side-by-side with the uncalibrated evaluation on the same 10
  cases, with clear before/after labeling.
- **SC-005**: Operators can complete at least one full cycle—scripted baseline
  eval, five MLflow UI overrides, align, scripted re-evaluate—in a single
  working session, and can start a second cycle without losing the original
  baseline results.
- **SC-006**: For the judge(s) aligned in the first round, aligned outcomes
  agree with the expert overrides on at least 4 of 5 overridden assessments
  after that alignment round.

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
  same required Markdown section headings.
- Assistant responses use fixed Markdown headings (Price context, News,
  Financial statements, Risks/limitations); section bodies are free text.
- Golden-suite market facts are validated live: at eval time, market data is
  re-fetched (or taken from the run’s tool outputs) and treated as ground
  truth for Groundedness; cases do not ship frozen price/statement snapshots
  as authoritative expected facts.
- Golden-suite tool expectations are a required unordered tool set per case;
  extra calls may still be scored by ToolCallEfficiency, but reordering alone
  is not a hard suite failure.
- ToolCallCorrectness covers appropriate tool choice and argument use;
  ToolCallEfficiency covers redundant calls and reasoning thrash;
  Groundedness covers faithfulness to tool-returned facts (fulfills
  constitution groundedness / numerical consistency intent).
- Initial demo guidance expects overrides often to target ToolCallEfficiency
  (redundant calls / thrash), but the operator chooses which judge(s) to
  align each round; no fixed default set.
- MemAlign `reflection_lm` will be pinned to a Gemini-family model;
  `embedding_model` will be explicitly pinned at implementation (no silent
  OpenAI default) per constitution.
- This feature is a local demo/showcase: no multi-user auth, no brokerage
  trading, and no private customer portfolio data.
- “Separate dataset experiment tags” means distinct evaluation-run tags /
  metadata separating uncalibrated vs aligned scoring while sharing the same
  `dataset_version`.
- Primary UX split: scripts run the agent and evaluations; the MLflow UI is
  the primary surface for reviewing traces and attaching human override
  annotations; alignment is invoked after annotations exist (typically by
  script).
