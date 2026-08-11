"""Readable stdio progress for agent runs and judge evaluations."""

from __future__ import annotations

import os
import shutil
import sys
import threading
from typing import Any

_LOCK = threading.Lock()
_ENABLED = os.getenv("EDD_QUIET", "").strip().lower() not in {"1", "true", "yes"}


def set_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = enabled


def enabled() -> bool:
    return _ENABLED


def _use_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


def cyan(text: str) -> str:
    return _c("36", text)


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def magenta(text: str) -> str:
    return _c("35", text)


def red(text: str) -> str:
    return _c("31", text)


def width() -> int:
    return max(48, min(100, shutil.get_terminal_size((80, 20)).columns))


def truncate(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _emit(line: str = "") -> None:
    if not _ENABLED:
        return
    with _LOCK:
        print(line, flush=True)


def rule(char: str = "─", label: str | None = None) -> None:
    w = width()
    if not label:
        _emit(dim(char * w))
        return
    pad = max(0, w - len(label) - 4)
    left = pad // 2
    right = pad - left
    _emit(dim(char * left) + f" {bold(label)} " + dim(char * right))


def banner(title: str, **meta: Any) -> None:
    rule("═", title)
    for key, value in meta.items():
        if value is None or value == "":
            continue
        _emit(f"  {dim(key + ':')} {value}")


def section(title: str) -> None:
    _emit()
    _emit(bold(cyan(f"▸ {title}")))


def step(kind: str, title: str, detail: str | None = None) -> None:
    colors = {
        "start": cyan,
        "model": magenta,
        "tool": yellow,
        "judge": magenta,
        "code": cyan,
        "ok": green,
        "fail": red,
        "info": dim,
        "result": green,
    }
    paint = colors.get(kind, lambda s: s)
    _emit(f"  {paint('●')} {bold(title)}")
    if detail:
        for line in str(detail).splitlines() or [""]:
            _emit(f"      {dim(line)}")


def kv(key: str, value: Any) -> None:
    _emit(f"  {dim(key + ':')} {value}")


def block(title: str, body: str, *, limit: int = 1200) -> None:
    _emit(f"  {dim(title)}")
    text = str(body or "").rstrip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    if not text:
        _emit(f"      {dim('(empty)')}")
        return
    for line in text.splitlines():
        _emit(f"      {line}")


def outcome(ok: bool, summary: str) -> None:
    mark = green("PASS") if ok else red("FAIL")
    _emit(f"  {bold('Outcome:')} {mark}  {summary}")


def feedback_value(value: Any) -> str:
    if value is True:
        return green("true")
    if value is False:
        return red("false")
    text = str(value)
    lowered = text.lower()
    if lowered in {"efficient", "correct", "grounded", "pass", "true"}:
        return green(text)
    if lowered in {"inefficient", "incorrect", "ungrounded", "fail", "false"}:
        return red(text)
    return yellow(text)


def extract_feedback_fields(fb: Any) -> tuple[Any, str | None, Any]:
    """Normalize MLflow Feedback / Assessment-like objects."""
    err = getattr(fb, "error", None)
    rationale = getattr(fb, "rationale", None)
    value = getattr(fb, "value", None)
    feedback = getattr(fb, "feedback", None)
    if feedback is not None and value is None:
        value = getattr(feedback, "value", None)
        if err is None:
            err = getattr(feedback, "error", None)
    return value, rationale, err
