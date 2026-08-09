# Contract: CLI / Orchestration Entrypoints

**Module**: `main.py` (orchestrator); helpers in other modules

## Commands (conceptual)

Exact argparse flags may vary; behavior MUST cover:

### `run-agent`

- **Input**: `--question` and/or `--ticker` (builds canonical analysis prompt)
- **Behavior**: Start MCP stdio tools, invoke LangGraph agent, print Markdown
- **Side effects**: Traces to `sqlite:///mlflow.db`

### `run-baseline-eval`

- **Input**: `--dataset-version` (default `v1`)
- **Behavior**: Ensure golden dataset exists; run `mlflow.genai.evaluate` with
  uncalibrated judges + code scorers; tag `eval_phase=uncalibrated`,
  `alignment_round=0`
- **Output**: Run id / summary of scores

### `seed-feedback` (demo)

- **Input**: path to feedback JSON (default `data/expert_feedback_seed.json`)
- **Behavior**: Attach ≥5 HUMAN assessments to named baseline traces
- **Note**: Mimics MLflow UI overrides; UI remains primary for real operators

### `align-and-reeval`

- **Input**: `--judges` list (operator-selected), `--alignment-round N`
- **Behavior**: Load traces with human assessments → MemAlign → `register` →
  evaluate/rescore with `eval_phase=aligned`
- **Invariants**: Does not delete uncalibrated baseline evidence

## Environment

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Judge + reflection LM |
| `MLFLOW_TRACKING_URI` | Override (default `sqlite:///mlflow.db`) |
| `LMSTUDIO_BASE_URL` | Default `http://localhost:1234/v1` |
| `AGENT_MODEL` | Default `qwen/qwen3.6-35b-a3b` |
| `OPENAI_API_KEY` | Only if embedding pin requires OpenAI |

## UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Used for trace review and human annotation (primary annotate path).
