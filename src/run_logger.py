"""Lightweight run logging for anomaly-detection scripts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_run_log(
    *,
    output_dir: Path | str,
    script_name: str,
    started_at_utc: str,
    ended_at_utc: str,
    duration_seconds: float,
    status: str,
    metrics: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run_log.jsonl"
    record = {
        "script": script_name,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": duration_seconds,
        "status": status,
        "metrics": metrics or {},
        "details": details or {},
        "error": error,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return log_path
