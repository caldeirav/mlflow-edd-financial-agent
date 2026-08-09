# Contract: Judges & Alignment

**Module**: `eval_pipeline.py`

## Qualitative judges

| Name | Model | Instructions focus |
|------|-------|--------------------|
| `ToolCallEfficiency` | `gemini:/gemini-2.5-pro` | Redundant tool calls / reasoning thrash on `{{ trace }}` |
| `ToolCallCorrectness` | `gemini:/gemini-2.5-pro` | Appropriate tools/args vs request + `{{ expectations }}` |
| `Groundedness` | `gemini:/gemini-2.5-pro` | Numeric/factual claims vs tool outputs on `{{ trace }}` |

**API**: `from mlflow.genai.judges import make_judge`

**Evaluate**: `mlflow.genai.evaluate(..., scorers=[...])`

## Code scorers (quantitative)

| Name | Rule |
|------|------|
| `RequiredMarkdownSections` | All expected headings present in output |
| `RequiredToolsUsed` | Every name in `expectations.required_tools` appears in trace tool spans |

## MemAlign

```text
optimizer = MemAlignOptimizer(
  reflection_lm="<pinned gemini flash-class URI>",
  embedding_model="<explicit pin from config>",
)
aligned = unaligned.align(feedback_traces, optimizer=optimizer)
aligned.register(experiment_id=...)
```

**Feedback traces**: Must include HUMAN assessments whose `name` equals the
judge name being aligned; include rationale text.

**Operator choice**: `--judges` selects subset to align each round.

## Run tagging

Uncalibrated and aligned evaluations MUST differ by `eval_phase` and
`judge_version` / `alignment_round` while sharing `dataset_version`.
