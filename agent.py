"""LangGraph ReAct financial assistant backed by LMStudio + MCP Yahoo tools."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import mlflow
from langchain_core.callbacks import BaseCallbackHandler
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import config
import console_trace as ct

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
# Serialize MCP stdio + asyncio.run across MLflow eval worker threads.
_AGENT_LOCK = threading.Lock()


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


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def _compact_tool_observation(name: str, text: str, *, max_chars: int = 6000) -> str:
    """Pack tool results for Gemini judges without blind mid-JSON truncation.

    Blind 1200-char cuts dropped later news articles (e.g. FedEx, earnings
    figures), causing false Groundedness failures. Prefer structured compaction.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        tool = name or ""
        if "news" in tool or "news" in parsed:
            articles = parsed.get("news") or []
            compact_news = []
            for article in articles:
                if not isinstance(article, dict):
                    continue
                compact_news.append(
                    {
                        "title": article.get("title"),
                        "summary": article.get("summary"),
                        "publisher": article.get("publisher"),
                    }
                )
            packed = json.dumps(
                {"ticker": parsed.get("ticker"), "news": compact_news},
                ensure_ascii=False,
            )
            return packed if len(packed) <= max_chars else packed[: max_chars - 1] + "…"

        if "statement" in tool or "statement_type" in parsed or "rows" in parsed:
            rows = parsed.get("rows") or {}
            # Keep all row labels; trim each series to first 4 periods (already typical).
            compact_rows: dict[str, Any] = {}
            for key, values in rows.items():
                if isinstance(values, list):
                    compact_rows[str(key)] = values[:4]
                else:
                    compact_rows[str(key)] = values
            packed = json.dumps(
                {
                    "ticker": parsed.get("ticker"),
                    "statement_type": parsed.get("statement_type"),
                    "columns": parsed.get("columns"),
                    "rows": compact_rows,
                },
                ensure_ascii=False,
            )
            if len(packed) <= max_chars:
                return packed
            # Last resort: drop half the rows, keep keys that look material.
            priority = (
                "Revenue",
                "Net Income",
                "EBITDA",
                "Free Cash Flow",
                "Total Debt",
                "Total Assets",
                "Operating Cash Flow",
                "Gross Profit",
            )
            slim: dict[str, Any] = {}
            for key in compact_rows:
                if any(p.lower() in key.lower() for p in priority):
                    slim[key] = compact_rows[key]
            if len(slim) < 8:
                for key, values in list(compact_rows.items())[:20]:
                    slim[key] = values
            packed = json.dumps(
                {
                    "ticker": parsed.get("ticker"),
                    "statement_type": parsed.get("statement_type"),
                    "columns": parsed.get("columns"),
                    "rows": slim,
                    "note": "rows compacted for judge context",
                },
                ensure_ascii=False,
            )
            return packed if len(packed) <= max_chars else packed[: max_chars - 1] + "…"

        if "price" in parsed or "get_stock_price" in tool:
            packed = json.dumps(parsed, ensure_ascii=False)
            return packed if len(packed) <= max_chars else packed[: max_chars - 1] + "…"

        packed = json.dumps(parsed, ensure_ascii=False)
        return packed if len(packed) <= max_chars else packed[: max_chars - 1] + "…"

    return raw if len(raw) <= max_chars else raw[: max_chars - 1] + "…"


def _tools_from_messages(messages: list[Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Extract tool call names and compacted tool observation text from LangChain messages."""
    called: list[str] = []
    observations: list[dict[str, str]] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                called.append(str(name))
        msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)
        name = getattr(msg, "name", None)
        if msg_type in ("tool", "ToolMessage") or name in config.REQUIRED_TOOLS_DEFAULT:
            text = _message_text(getattr(msg, "content", ""))
            observations.append(
                {
                    "name": str(name or "tool"),
                    "content": _compact_tool_observation(str(name or "tool"), text),
                }
            )
    return called, observations


def tools_from_trace_spans(spans: list[Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Rebuild compacted tool observations from MLflow span outputs (full tool text)."""
    called: list[str] = []
    observations: list[dict[str, str]] = []
    for span in spans or []:
        name = str(getattr(span, "name", "") or "")
        if name not in config.REQUIRED_TOOLS_DEFAULT:
            continue
        called.append(name)
        raw_out = getattr(span, "outputs", None)
        text = ""
        if isinstance(raw_out, dict):
            content = raw_out.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    text = str(first.get("text") or "")
                else:
                    text = json.dumps(content, default=str)
            else:
                text = json.dumps(raw_out, default=str)
        elif raw_out is not None:
            text = _message_text(raw_out)
        observations.append(
            {
                "name": name,
                "content": _compact_tool_observation(name, text),
            }
        )
    return called, observations


def _fallback_report(reason: str) -> str:
    return (
        "## Price context\n\nUnavailable.\n\n"
        "## News\n\nUnavailable.\n\n"
        "## Financial statements\n\nUnavailable.\n\n"
        f"## Risks/limitations\n\n{reason}\n"
    )


def _pack_result(
    report: str,
    tools_called: list[str] | None = None,
    tool_observations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "report": report,
        "tools_called": list(tools_called or []),
        "tool_observations": list(tool_observations or []),
    }


def _fmt_tool_args(args: Any) -> str:
    try:
        if isinstance(args, (dict, list)):
            return ct.truncate(json.dumps(args, ensure_ascii=False), 180)
        return ct.truncate(str(args), 180)
    except Exception:
        return ct.truncate(str(args), 180)


class _ConsoleCallback(BaseCallbackHandler):
    """Stream model/tool activity to stdio during an agent run."""

    def __init__(self) -> None:
        super().__init__()
        self.model_n = 0
        self.tool_n = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.model_n += 1
        ct.step("model", f"Model call #{self.model_n}", config.AGENT_MODEL)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        gens = getattr(response, "generations", None) or []
        content = ""
        tool_names: list[str] = []
        if gens and gens[0]:
            gen = gens[0][0]
            message = getattr(gen, "message", None)
            if message is not None:
                content = _message_text(getattr(message, "content", "") or "")
                for tc in getattr(message, "tool_calls", None) or []:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name:
                        tool_names.append(str(name))
            elif getattr(gen, "text", None):
                content = str(gen.text)
        if tool_names:
            ct.step(
                "model",
                f"Model response #{self.model_n}",
                "requests tools: " + ", ".join(tool_names),
            )
        elif content.strip():
            ct.step(
                "model",
                f"Model response #{self.model_n}",
                ct.truncate(content, 220),
            )
        else:
            ct.step("model", f"Model response #{self.model_n}", "(no text)")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.tool_n += 1
        name = (
            (serialized or {}).get("name")
            or kwargs.get("name")
            or (inputs or {}).get("name")
            or "tool"
        )
        args = inputs if inputs is not None else input_str
        ct.step("tool", f"Tool call #{self.tool_n}: {name}", f"args={_fmt_tool_args(args)}")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name = kwargs.get("name") or "tool"
        text = _message_text(getattr(output, "content", output))
        ct.step("tool", f"Tool result: {name}", ct.truncate(text, 260))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        ct.step("fail", "Tool error", f"{type(error).__name__}: {error}")


def _print_agent_result(result: dict[str, Any], started: float) -> None:
    elapsed = time.perf_counter() - started
    ct.section("Final result")
    tools = result.get("tools_called") or []
    ct.kv("tools_called", ", ".join(tools) if tools else ct.dim("(none)"))
    ct.kv("tool_observations", str(len(result.get("tool_observations") or [])))
    ct.kv("elapsed", f"{elapsed:.1f}s")
    ct.block("report", result.get("report") or "", limit=900)
    ct.rule("═")


async def _run_agent_async(question: str) -> dict[str, Any]:
    started = time.perf_counter()
    ct.banner(
        "AGENT RUN",
        model=config.AGENT_MODEL,
        endpoint=config.LMSTUDIO_BASE_URL,
    )
    ct.section("Query")
    ct.block("user", question, limit=500)

    mlflow.langchain.autolog()
    llm = build_llm()
    ct.section("Setup")
    ct.step("start", "Loading MCP tools (Yahoo Finance)")
    client = MultiServerMCPClient(_mcp_server_command())
    tools = await client.get_tools()
    if not tools:
        result = _pack_result(_fallback_report("Local market-data tools failed to load."))
        ct.step("fail", "No MCP tools loaded")
        _print_agent_result(result, started)
        return result

    tool_names = [getattr(t, "name", str(t)) for t in tools]
    ct.step("ok", f"Tools ready ({len(tool_names)})", ", ".join(tool_names))

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    ct.section("Execution")
    callback = _ConsoleCallback()
    result_state = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [callback]},
    )
    messages = list(result_state.get("messages") or [])
    if not messages:
        result = _pack_result(_fallback_report("Agent returned no messages."))
        _print_agent_result(result, started)
        return result

    tools_called, tool_observations = _tools_from_messages(messages)
    last = messages[-1]
    report = _message_text(getattr(last, "content", last))
    result = _pack_result(report, tools_called, tool_observations)
    _print_agent_result(result, started)
    return result


async def _run_agent_async_safe(question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        return await _run_agent_async(question)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if name in {"ConnectionError", "APIConnectionError"} or "Connection" in name:
            reason = (
                f"LMStudio unreachable at {config.LMSTUDIO_BASE_URL} ({name}: {exc}). "
                "Start the local server and ensure AGENT_MODEL is loaded."
            )
            ct.step("fail", "LMStudio connection failed", str(exc))
        else:
            reason = f"Agent/tool failure ({name}): {exc}"
            ct.step("fail", f"Agent failed ({name})", str(exc))
        result = _pack_result(_fallback_report(reason))
        _print_agent_result(result, started)
        return result


def run_analysis(question: str) -> dict[str, Any]:
    """Run the financial assistant synchronously. Never fabricates tool data.

    Returns a dict with report Markdown plus tools_called / tool_observations so
    Gemini judges can score without {{ trace }} (unsupported with JSON mime type).
    """
    with _AGENT_LOCK:
        return asyncio.run(_run_agent_async_safe(question))


def predict_fn(question: str, ticker: str | None = None) -> dict[str, Any]:
    """predict_fn compatible with mlflow.genai.evaluate inputs."""
    if not question and ticker:
        question = config.build_analysis_question(ticker)
    return run_analysis(question)
