"""FastMCP server wrapping Yahoo Finance (yfinance) tools.

Tools: get_stock_price, get_financial_news, get_financial_statements.
Transport: stdio (default) for LangGraph MCP adapters.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("yahoo-finance")


def _safe_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker.strip().upper())


@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """Return the latest stock price and short context for a ticker."""
    try:
        t = _safe_ticker(ticker)
        info: dict[str, Any] = {}
        try:
            info = t.fast_info.__dict__ if hasattr(t, "fast_info") else {}
        except Exception:
            info = {}
        # Prefer fast_info fields; fall back to history
        price = None
        currency = None
        if info:
            price = info.get("last_price") or info.get("lastPrice")
            currency = info.get("currency")
        if price is None:
            hist = t.history(period="5d")
            if hist is None or hist.empty:
                return json.dumps(
                    {"error": f"No price data for ticker '{ticker}'", "ticker": ticker}
                )
            price = float(hist["Close"].iloc[-1])
        meta = {
            "ticker": ticker.upper(),
            "price": float(price) if price is not None else None,
            "currency": currency,
        }
        try:
            long_name = getattr(t, "info", {}) or {}
            if isinstance(long_name, dict):
                meta["name"] = long_name.get("shortName") or long_name.get("longName")
        except Exception:
            pass
        return json.dumps(meta, default=str)
    except Exception as exc:  # noqa: BLE001 — surface tool errors to the agent
        return json.dumps({"error": str(exc), "ticker": ticker})


@mcp.tool()
def get_financial_news(ticker: str, limit: int = 5) -> str:
    """Return recent financial news headlines for a ticker."""
    try:
        t = _safe_ticker(ticker)
        news = getattr(t, "news", None) or []
        items = []
        for entry in news[: max(1, min(limit, 20))]:
            content = entry.get("content") if isinstance(entry, dict) else None
            if isinstance(content, dict):
                items.append(
                    {
                        "title": content.get("title"),
                        "summary": content.get("summary"),
                        "publisher": (content.get("provider") or {}).get("displayName")
                        if isinstance(content.get("provider"), dict)
                        else None,
                        "link": (content.get("canonicalUrl") or {}).get("url")
                        if isinstance(content.get("canonicalUrl"), dict)
                        else None,
                    }
                )
            elif isinstance(entry, dict):
                items.append(
                    {
                        "title": entry.get("title"),
                        "publisher": entry.get("publisher"),
                        "link": entry.get("link"),
                    }
                )
        if not items:
            return json.dumps(
                {"ticker": ticker.upper(), "news": [], "note": "No news returned"}
            )
        return json.dumps({"ticker": ticker.upper(), "news": items}, default=str)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc), "ticker": ticker})


@mcp.tool()
def get_financial_statements(
    ticker: str,
    statement_type: Literal["income", "balance", "cashflow"] = "income",
) -> str:
    """Return key financial statement rows for a ticker (truncated)."""
    try:
        t = _safe_ticker(ticker)
        frame = None
        if statement_type == "income":
            frame = t.income_stmt
        elif statement_type == "balance":
            frame = t.balance_sheet
        else:
            frame = t.cashflow
        if frame is None or getattr(frame, "empty", True):
            return json.dumps(
                {
                    "error": f"No {statement_type} statement for '{ticker}'",
                    "ticker": ticker,
                }
            )
        # Keep a compact slice for LLM context
        head = frame.iloc[:12, :4]
        payload = {
            "ticker": ticker.upper(),
            "statement_type": statement_type,
            "columns": [str(c) for c in head.columns],
            "rows": {
                str(idx): [None if v != v else v for v in row.tolist()]  # NaN -> None
                for idx, row in head.iterrows()
            },
        }
        return json.dumps(payload, default=str)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc), "ticker": ticker})


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
