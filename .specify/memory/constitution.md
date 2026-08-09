<!--
Sync Impact Report
- Version change: 1.0.0 → 1.1.0
- Modified principles:
  I. Persistent Agent Tracing → expanded (autolog, span types, assessments, UI)
  II. Model-as-a-Judge (Gemini) → redefined (make_judge + gemini:/ URI; not raw SDK-only)
  III. Judge Alignment (MemAlign) → redefined (HITL prerequisites, register/unalign, reduce FPs)
  IV. Tracing Invariants & Golden Suite → expanded (eval datasets, baselines, version tags)
- Added sections:
  V. Change-Control & Regression Gates
  VI. Experiment Taxonomy & Model Roles
  VII. Evaluation Data, Privacy & Cost Hygiene
  (workflow rewritten to MLflow judge cycle)
- Removed sections: none
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/templates/commands/*.md ⚠ pending (directory absent)
  - README.md ✅ updated (make_judge, datasets, regression gates)
- Follow-up TODOs:
  - Pin concrete MemAlign embedding_model URI when implementation chooses provider
-->

# MLflow EDD Financial Agent Constitution

## Core Principles

### I. Persistent Agent Tracing

All agent and tool invocations MUST produce durable MLflow traces in the local
tracking store at `sqlite:///mlflow.db`. Configuration MUST call
`mlflow.set_tracking_uri("sqlite:///mlflow.db")` and use a named experiment
before runs. LangGraph/LangChain execution MUST enable
`mlflow.langchain.autolog()` (or equivalent auto-tracing) so the graph is
captured end-to-end.

Traces MUST include appropriate span types for the execution graph
(`AGENT`, `TOOL`, `LLM` as applicable). Qualitative judge results MUST be
recorded as MLflow assessments on traces (not only informal logs). Runs
without durable traces MUST NOT be accepted as experiment evidence. Local UI
comparison MUST use
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

**Rationale**: Persistent, typed spans plus assessments are the audit trail for
tool usage and financial reasoning and the substrate for MemAlign and UI
side-by-side comparison.

### II. Model-as-a-Judge (Gemini)

All qualitative evaluations MUST use Google Gemini `gemini-2.5-pro` as the
judge model through MLflow's GenAI judge/scorer APIs—primarily
`mlflow.genai.judges.make_judge` with model URI `gemini:/gemini-2.5-pro`
(or the equivalent supported Gemini provider URI)—and MUST be executed via
`mlflow.genai.evaluate` (or equivalent scorer invocation that attaches
assessments to traces).

Judges MUST be named scorers with an explicit `feedback_value_type` and
instructions that use supported template variables (`{{ inputs }}`,
`{{ outputs }}`, `{{ expectations }}`, `{{ trace }}` as appropriate). At
minimum the project MUST define scorers for:

- `tool_usage_efficiency`
- `financial_reasoning`
- financial groundedness / numerical consistency with tool outputs (or
  citation-to-tool-output)

Code-based (quantitative) scorers are allowed alongside LLM judges. Ad-hoc
free-text scoring outside MLflow scorers, or alternate qualitative judge
models, MUST NOT be used. Direct `google-genai` calls MUST NOT replace
`make_judge` / evaluate for official qualitative metrics (the SDK may exist
only as a transitive/provider detail under MLflow).

**Rationale**: MLflow-native judges keep scores, registration, and alignment on
one surface; a pinned Gemini model keeps qualitative criteria comparable.

### III. Judge Alignment (MemAlign)

When qualitative judges disagree with human judgment on tool usage efficiency
or financial reasoning (false positives/negatives), the system MUST run an
alignment loop using MLflow's `MemAlignOptimizer` (experimental; treat API
churn as expected and pin MLflow versions in the lockfile).

Alignment MUST:

1. Collect human assessments on traces with the **same name** as the judge,
   including natural-language rationale, with a mix of positive and negative
   labels (recommend ≥10 labeled traces before aligning).
2. Call `judge.align(traces=..., optimizer=MemAlignOptimizer(...))`.
3. `register()` the aligned judge and record `judge_version` / alignment
   metadata on subsequent eval runs.
4. Use `unalign(...)` when feedback was wrong or requirements changed.

MemAlign reduces—not guarantees elimination of—disagreement. Shipping an
updated judge after known systematic false positives without a documented
align + register pass is non-compliant. `reflection_lm` and `embedding_model`
MUST be explicitly pinned (see Technology Stack); silent reliance on
MemAlign's OpenAI embedding default is forbidden without Complexity Tracking
justification.

**Rationale**: Human-labeled traces are the MemAlign contract; registered
aligned judges keep calibration reproducible across experiments.

### IV. Tracing Invariants & Golden Suite

The project MUST maintain a versioned MLflow evaluation dataset (golden suite)
with expectations / ground-truth fields where checkable (expected answers,
required tool invariants). Every agent change under test MUST run
`mlflow.genai.evaluate` (or equivalent) against a declared dataset version and
persist traces for MLflow UI side-by-side comparison of the **same dataset**
across agent and/or judge versions.

Progressive improvement MUST bump `dataset_version` when:

- eval/run failures identify gaps, or
- human feedback on agent traces is accepted into the suite.

Deleting or skipping golden-suite capture for a claimed experiment is
forbidden. Expectations MUST be preferred over judge-only checks for
verifiable financial facts.

**Rationale**: Versioned datasets plus expectations are the regression
invariant; human feedback grows both the suite and judge alignment inputs.

### V. Change-Control & Regression Gates

Before merging agent, prompt, tool, or judge changes that affect behavior:

1. Run the declared golden dataset against the candidate.
2. Compare results to a **frozen baseline** run (same `dataset_version`).
3. Fail the gate on regressions versus baseline (not only on absolute scores).

Features MUST NOT be marked done without durable traces, scorer assessments
where qualitative eval applies, baseline comparison evidence, and tags that
identify what was evaluated (Principle VI).

**Rationale**: Absolute scores drift with judges and data; regressions against
a frozen baseline are the delivery gate.

### VI. Experiment Taxonomy & Model Roles

Every eval/experiment run MUST tag (or otherwise record) at least:

- `agent_version`
- `judge_version`
- `dataset_version`
- `alignment_round` (or equivalent when alignment is involved)

Model roles MUST stay separated:

| Role | Required choice |
|------|-----------------|
| Agent inference | Qwen 3.6 35B via LMStudio |
| Qualitative judge | Gemini `gemini-2.5-pro` via MLflow URI |
| MemAlign `reflection_lm` | Explicitly pinned; SHOULD be Gemini family |
| MemAlign `embedding_model` | Explicitly pinned in project config |

Judges MUST NOT be silently swapped mid-experiment. Promote baseline vs
aligned judges deliberately and record which `judge_version` scored each run.

**Rationale**: Side-by-side UI comparison is meaningless without version tags
and role separation.

### VII. Evaluation Data, Privacy & Cost Hygiene

Human feedback is first-class: corrections MUST be stored as MLflow
assessments with a human source (e.g. `AssessmentSourceType.HUMAN`) on the
relevant traces so they can feed MemAlign and suite growth.

Plans and code MUST document which market/user payloads may leave the local
machine to Gemini (judge/reflection) and MUST minimize unnecessary PII or
raw account data in judge prompts. Prefer cheaper Gemini variants for
`reflection_lm` when alignment quality remains acceptable; judge scoring for
official gates remains `gemini-2.5-pro`.

**Rationale**: Feedback quality and data boundaries are part of EDD, not
optional ops notes.

## Technology Stack Constraints

- **Runtime**: Python `>=3.13` with declared project dependencies + lockfile.
- **Agent orchestration**: LangGraph; enable `mlflow.langchain.autolog()`.
- **Local agent model**: Qwen 3.6 35B via LMStudio.
- **Tools**: MCP Yahoo Finance (`mcp`, `yfinance`).
- **Observability & EDD**: MLflow `>=3.15.1`; tracking URI `sqlite:///mlflow.db`;
  UI via `--backend-store-uri sqlite:///mlflow.db`.
- **Qualitative judge**: `make_judge` + `gemini:/gemini-2.5-pro` (Gemini URI);
  run through `mlflow.genai.evaluate` / scorers.
- **Alignment**: `MemAlignOptimizer` with pinned `reflection_lm` and
  `embedding_model`; extras as required by MLflow (`dspy`, `jinja2`, `tqdm`).
- **Tests**: `pytest` plus MLflow golden-dataset evaluation for regressions.

Deviations require a constitution amendment and a Complexity Tracking entry.

## Evaluation-Driven Development Workflow

1. **Instrument**: Set tracking URI + experiment; enable LangGraph autolog.
2. **Define dataset**: Versioned golden cases with expectations where possible.
3. **Evaluate**: `mlflow.genai.evaluate` with named Gemini judges and any
   code scorers; tag `agent_version` / `judge_version` / `dataset_version`.
4. **Collect human feedback**: Human assessments + rationale on disagreeing
   traces.
5. **Align when warranted**: `MemAlignOptimizer` → `align` → `register`; bump
   `alignment_round` / `judge_version`; use `unalign` if feedback was bad.
6. **Grow suite**: Failures and accepted feedback → new `dataset_version`.
7. **Gate**: Compare to frozen baseline on the same dataset; block merge on
   regressions.

## Governance

This constitution supersedes conflicting informal practices for this
repository. Amendments MUST:

1. Update `.specify/memory/constitution.md` with a semantic version bump:
   - **MAJOR**: Remove or redefine a principle incompatibly.
   - **MINOR**: Add a principle/section or materially expand guidance.
   - **PATCH**: Clarifications, wording, and non-semantic refinements.
2. Set **Last Amended** to the amendment date (ISO `YYYY-MM-DD`).
3. Propagate changes to dependent templates (plan, spec, tasks) and note
   impact in the Sync Impact Report comment at the top of this file.
4. Require PR / review compliance checks against the Constitution Check
   gates in the implementation plan template.

Complexity that violates these principles MUST be justified in the plan's
Complexity Tracking table or the work MUST be redesigned. Runtime
development guidance follows the active feature `plan.md` under `specs/`.

**Version**: 1.1.0 | **Ratified**: 2026-08-09 | **Last Amended**: 2026-08-09
