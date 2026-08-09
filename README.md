# mlflow-edd-financial-agent

A financial AI agent built with LangGraph and local Qwen 3.6 35B (LMStudio), utilizing MCP Yahoo Finance tools. Showcases MLflow Evaluation-Driven Development (EDD) with:

- Persistent SQLite tracing (`sqlite:///mlflow.db`) and LangGraph autolog
- Gemini LLM judges via MLflow `make_judge` (`gemini:/gemini-2.5-pro`) and `mlflow.genai.evaluate`
- Expert judge alignment (MemAlign) with registered judge versions
- Versioned golden evaluation datasets, baseline regression gates, and side-by-side trace comparison before vs after calibration

Project governance: `.specify/memory/constitution.md` (v1.1.0).
