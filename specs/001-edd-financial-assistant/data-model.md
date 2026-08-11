# Data Model: EDD Financial Assistant

**Branch**: `001-edd-financial-assistant` | **Date**: 2026-08-09

## Entities

### AnalysisRequest

| Field | Type | Notes |
|-------|------|-------|
| question | string | Canonical financial analysis prompt |
| ticker | string | Yahoo Finance symbol (e.g. AAPL) |
| focus | string? | Optional light variation (news-heavy, statements-heavy) |

**Validation**: ticker non-empty; question includes analysis intent.

### AssistantResponse

| Field | Type | Notes |
|-------|------|-------|
| markdown | string | Full response body |
| sections | map[string→string] | Parsed by heading |

**Required headings** (exact checklist):

1. `Price context`
2. `News`
3. `Financial statements`
4. `Risks/limitations`

**Validation**: All four headings present (case-sensitive match per scorer
config); bodies may be free text including “unavailable” notes.

### MarketDataObservation

| Field | Type | Notes |
|-------|------|-------|
| tool_name | string | `get_stock_price` \| `get_financial_news` \| `get_financial_statements` |
| ticker | string | |
| payload | JSON/text | Live yfinance-derived content |
| captured_at | datetime | Trace span timestamp |

**Relationship**: Produced by MCP tools; Groundedness compares claims in
`AssistantResponse` primarily to observations on the **same** ExecutionTrace.

### GoldenCase

| Field | Type | Notes |
|-------|------|-------|
| case_id | string | Stable id `case-01`…`case-10` |
| inputs | AnalysisRequest | |
| required_sections | string[] | Usually the four Markdown headings |
| required_tools | string[] | Unordered set; missing any → hard fail |
| groundedness_policy | enum | Always `live_tool_outputs` for v1 |
| dataset_version | string | e.g. `v1` |

**Validation**: Exactly 10 cases in `dataset_version=v1`; required_tools
non-empty.

### ExecutionTrace

| Field | Type | Notes |
|-------|------|-------|
| trace_id | string | MLflow trace id |
| case_id | string? | Linked golden case when from suite |
| spans | AGENT/TOOL/LLM | Via autolog |
| tags | map | agent_version, dataset_version, … |

**Lifecycle**: created on agent invoke → assessments attached → optionally
merged into eval dataset.

### JudgeAssessment

| Field | Type | Notes |
|-------|------|-------|
| name | string | Must match judge: ToolCallEfficiency, ToolCallCorrectness, Groundedness |
| value | typed | Per `feedback_value_type` |
| rationale | string | |
| source | JUDGE \| HUMAN | |
| judge_version | string | |

### HumanOverride

| Field | Type | Notes |
|-------|------|-------|
| trace_id | string | |
| judge_name | string | Target scorer for this override |
| value | typed | Expert label |
| rationale | string | Required for MemAlign quality |
| origin | `mlflow_ui` \| `seed_dict` | |

**Rule**: ≥5 overrides per alignment round for the selected judge(s); names
must match judges being aligned.

### AlignmentRound

| Field | Type | Notes |
|-------|------|-------|
| alignment_round | int | 0 = uncalibrated baseline |
| selected_judges | string[] | Operator choice |
| unaligned_judge_version | string | |
| aligned_judge_version | string | After register |
| eval_phase | `uncalibrated` \| `aligned` | Run tag |

### EvaluationRunTags

Required tags on eval runs:

- `agent_version`
- `judge_version`
- `dataset_version`
- `alignment_round`
- `eval_phase` (`uncalibrated` \| `aligned`)

## Relationships

```text
GoldenCase (10) ──evaluates──► AnalysisRequest
     │
     ▼
ExecutionTrace ──produces──► AssistantResponse
     │                └──uses──► MarketDataObservation(s)
     │
     ├── JudgeAssessment (uncalibrated)
     ├── HumanOverride (5+)
     └── JudgeAssessment (aligned, after MemAlign)
```

## State transitions

1. **Dataset**: `draft` → `v1 registered` → (`v2+` on suite growth)
2. **Judge**: `uncalibrated` → `aligned_round_N` (register) → optional `unalign`
3. **Eval evidence**: `baseline_traces` → `uncalibrated_scores` →
   `human_overrides` → `aligned_scores` (baseline retained)
