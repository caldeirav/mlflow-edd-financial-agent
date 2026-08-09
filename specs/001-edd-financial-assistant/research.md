# Research: EDD Financial Assistant

**Branch**: `001-edd-financial-assistant` | **Date**: 2026-08-09

## 1. Module layout & orchestration

**Decision**: Implement five root modules — `mcp_server.py`, `agent.py`,
`golden_dataset.py`, `eval_pipeline.py`, `main.py` — plus thin `config.py`
for tracking URI, experiment name, model pins, and version tags.

**Rationale**: Matches the requested architecture; keeps EDD demo navigable
without a premature package split.

**Alternatives considered**: `src/` package layout (clearer for large apps,
heavier for this showcase); notebook-first (rejected by clarification D).

## 2. FastMCP + yfinance tools

**Decision**: Use FastMCP (`mcp.server.fastmcp` or `fastmcp`) in
`mcp_server.py` exposing at least:

- `get_stock_price(ticker)` — last/regular market price + basic context
- `get_financial_news(ticker)` — recent headlines/summaries
- `get_financial_statements(ticker, statement_type?)` — income/balance/cash
  (user sketch may say `get_financial_statements`; keep that exact tool name)

Transport: **stdio** for LangGraph subprocess embedding.

**Rationale**: Spec FR-002; yfinance needs no API key; stdio is the standard
MCP embedding path for local agents.

**Alternatives considered**: Direct `@tool` functions without MCP (simpler but
breaks “FastMCP server” requirement); HTTP MCP (extra process management).

## 3. LangGraph agent ↔ MCP + LMStudio

**Decision**: `agent.py` builds a ReAct-style LangGraph agent with
`ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
model="qwen/qwen3.6-35b-a3b")` (or config override). Load tools via
`langchain-mcp-adapters` `MultiServerMCPClient` pointing at
`python mcp_server.py` stdio. Enable `mlflow.langchain.autolog()` before
invoke. System prompt requires Markdown headings: Price context, News,
Financial statements, Risks/limitations.

**Rationale**: Constitution stack + clarifications; adapters are the supported
LangChain/LangGraph MCP bridge.

**Alternatives considered**: Manual MCP JSON-RPC client (more code); binding
tools without LangGraph (fails constitution).

**Dependency add**: `langchain-mcp-adapters` (and `fastmcp` if not using
`mcp.server.fastmcp`).

## 4. Tracing & experiment hygiene

**Decision**: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`; experiment
name e.g. `edd-financial-assistant`. Tag every eval run with
`agent_version`, `judge_version`, `dataset_version`, `alignment_round`, plus
`eval_phase=uncalibrated|aligned`.

**Rationale**: Constitution I + VI; enables UI side-by-side.

**Alternatives considered**: File store / remote tracking (overkill locally).

## 5. Golden dataset & expectations

**Decision**: `golden_dataset.py` creates/upserts MLflow GenAI dataset
`financial_analysis_golden` with `dataset_version=v1` (10 records). Each
record:

- `inputs`: `{ "question": <canonical template with ticker>, "ticker": ... }`
- `expectations`:
  - `required_sections`: list of Markdown headings
  - `required_tools`: unordered tool name set
  - `groundedness_policy`: `"live_tool_outputs"` (facts from trace tool
    spans / live yfinance at score time — not frozen prices)

Also support `mlflow.log_expectation` on traces when wiring annotated
datasets. Code scorers assert required sections + required tools; Gemini
judges handle qualitative axes.

**Rationale**: Clarifications A→live facts, B→required tool set; FR-004/005;
MLflow `create_dataset` / `merge_records` pattern.

**Alternatives considered**: Frozen CSV snapshots (rejected in clarify);
exact call sequences (rejected).

## 6. Judges (`make_judge`)

**Decision**: Import from `mlflow.genai.judges.make_judge` (not a fictional
`mlflow.genai.make_judge` shorthand in docs — use the judges submodule).
Model URI: `gemini:/gemini-2.5-pro`. Three judges:

| Name | Focus | Template vars |
|------|--------|---------------|
| `ToolCallEfficiency` | Redundant calls / reasoning thrash | `{{ trace }}` |
| `ToolCallCorrectness` | Right tools/args vs request + expectations | `{{ trace }}`, `{{ expectations }}` |
| `Groundedness` | Claims vs tool outputs on the trace | `{{ trace }}`, `{{ outputs }}` |

Use explicit `feedback_value_type` (e.g. Literal pass/fail or ordinal).

**Rationale**: Spec FR-007; constitution II; trace-based judges for agents.

**Alternatives considered**: Raw `google-genai` scoring (constitution forbids
as official path).

## 7. Human feedback & MemAlign

**Decision**:

1. **Primary (spec)**: Operator annotates in MLflow UI; assessments must use
   the same `name` as the judge being aligned.
2. **Demo seed (Complexity Tracking)**: `eval_pipeline.py` accepts an expert
   feedback dictionary / `data/expert_feedback_seed.json` that writes
   `AssessmentSourceType.HUMAN` assessments onto selected baseline traces
   (mimics UI overrides) so `main.py` can run align end-to-end.
3. Operator selects which judge(s) to align per round (CLI flag /
   config list).
4. `MemAlignOptimizer(reflection_lm="gemini:/gemini-2.5-flash",
   embedding_model=<PINNED>)` then
   `aligned = judge.align(traces, optimizer=...); aligned.register(...)`.
5. If Gemini embeddings are unsupported by MemAlign at implement time, pin
   `openai:/text-embedding-3-small` explicitly in `config.py` (Complexity
   Tracking row) — never rely on implicit default.

**Rationale**: Clarification D (UI) + user module sketch (feedback dict) +
constitution III/VII.

**Alternatives considered**: UI-only (blocks reproducible demo); aligning all
three judges always (rejected — operator chooses).

## 8. Evaluate pre/post alignment

**Decision**: `mlflow.genai.evaluate(data=dataset_or_traces, predict_fn=...,
scorers=[...])` for uncalibrated run (`alignment_round=0`,
`eval_phase=uncalibrated`). After align, either re-score stored baseline
traces with aligned scorers or re-run evaluate with aligned judges tagged
`eval_phase=aligned`, `alignment_round=N`. Prefer **same baseline traces**
for judge-only rescoring when API allows; otherwise re-invoke predict_fn but
keep identical `dataset_version` and compare score deltas.

**Rationale**: Spec FR-010; constitution V.

**Alternatives considered**: Only logging custom metrics without evaluate
(weaker UI integration).

## 9. Privacy / payloads to Gemini

**Decision**: Send only public ticker questions, assistant Markdown outputs,
and tool span summaries needed for judging. No account IDs or portfolios.

**Rationale**: Constitution VII / EDD-009.

## 10. Resolved unknowns

| Topic | Resolution |
|-------|------------|
| Exact financial statements tool name | `get_financial_statements` |
| make_judge import | `mlflow.genai.judges.make_judge` |
| Embedding pin | Explicit in config; prefer Gemini embedding URI if supported else documented OpenAI pin |
| Feedback path | UI primary + seeded dict for demos |
| Required deps to add | `langchain-mcp-adapters`, MemAlign extras (`dspy`, `jinja2`, `tqdm`), optionally `fastmcp` |
