"""LangGraph ReAct financial assistant backed by LMStudio + MCP Yahoo tools."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import mlflow
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import config

SYSTEM_PROMPT = """You are a financial analysis assistant.
Use the available market-data tools for prices, news, and financial statements.
Never invent numeric market facts. If a tool fails or data is missing, say so
clearly under the relevant section.

You MUST respond in Markdown with exactly these section headings (in order):

## Price context
## News
## Financial statements
## Risks/limitations

Keep section bodies concise and cite tool-derived figures when available.
"""

_ROOT = Path(__file__).resolve().parent


def _mcp_server_command() -> dict[str, Any]:
    return {
        "yahoo-finance": {
            "command": sys.executable,
            "args": [str(_ROOT / "mcp_server.py")],
            "transport": "stdio",
        }
    }


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.LMSTUDIO_BASE_URL,
        api_key=config.LMSTUDIO_API_KEY,
        model=config.AGENT_MODEL,
        temperature=0.2,
    )


async def _run_agent_async(question: str) -> str:
    mlflow.langchain.autolog()
    llm = build_llm()
    client = MultiServerMCPClient(_mcp_server_command())
    tools = await client.get_tools()
    if not tools:
        return (
            "## Price context\n\nUnavailable: no MCP tools loaded.\n\n"
            "## News\n\nUnavailable.\n\n"
            "## Financial statements\n\nUnavailable.\n\n"
            "## Risks/limitations\n\nLocal market-data tools failed to load.\n"
        )
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages") or []
    if not messages:
        return (
            "## Price context\n\nUnavailable: empty agent response.\n\n"
            "## News\n\nUnavailable.\n\n"
            "## Financial statements\n\nUnavailable.\n\n"
            "## Risks/limitations\n\nAgent returned no messages.\n"
        )
    last = messages[-1]
    content = getattr(last, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(last)


def run_analysis(question: str) -> str:
    """Run the financial assistant synchronously. Never fabricates tool data."""
    try:
        return asyncio.run(_run_agent_async(question))
    except ConnectionError as exc:
        return (
            "## Price context\n\nUnavailable: cannot reach local model "
            f"({config.LMSTUDIO_BASE_URL}).\n\n"
            "## News\n\nUnavailable.\n\n"
            "## Financial statements\n\nUnavailable.\n\n"
            f"## Risks/limitations\n\nLMStudio connection error: {exc}\n"
        )
    except Exception as exc:  # noqa: BLE001 — surface actionable failure to caller
        return (
            "## Price context\n\nUnavailable due to agent error.\n\n"
            "## News\n\nUnavailable.\n\n"
            "## Financial statements\n\nUnavailable.\n\n"
            f"## Risks/limitations\n\nAgent/tool failure: {exc}\n"
        )


def predict_fn(question: str, ticker: str | None = None) -> str:
    """predict_fn compatible with mlflow.genai.evaluate inputs."""
    if not question and ticker:
        question = config.build_analysis_question(ticker)
    return run_analysis(question)
