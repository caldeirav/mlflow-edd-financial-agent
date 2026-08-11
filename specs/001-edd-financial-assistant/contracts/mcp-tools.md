# Contract: MCP Yahoo Finance Tools

**Module**: `mcp_server.py`  
**Transport**: stdio (FastMCP)

## Tools

### `get_stock_price`

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| ticker | string | yes | Yahoo symbol |

**Returns**: Text/JSON with price and short context (currency, as-of if available).

**Errors**: Invalid ticker → structured error string (no throw that crashes server).

### `get_financial_news`

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| ticker | string | yes | Yahoo symbol |
| limit | int | no | Max articles (default 5) |

**Returns**: List of headline/summary/date items as text or JSON.

### `get_financial_statements`

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| ticker | string | yes | Yahoo symbol |
| statement_type | string | no | `income` \| `balance` \| `cashflow` (default `income`) |

**Returns**: Key statement lines suitable for LLM consumption (truncated if large).

## Invariants

- Tools MUST use `yfinance` live data (no fabricated numbers).
- Tool names above are the **required_tools** vocabulary for golden cases.
- Server MUST be runnable as: `python mcp_server.py` (stdio).
