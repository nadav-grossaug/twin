"""Logging that excludes sensitive observations.

Every release-layer logger goes through `RedactingFilter`, which drops
node ids and feature-instance keys from messages. Code should log counts
and stage names only; the filter is the backstop, not the primary control.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PATTERNS = [
    (re.compile(r"\bego_\d+__f\d+[A-Za-z0-9_]*\b"), "<feat>"),                 # ego-scoped feature keys
    (re.compile(r"\banonymized[ _]feature[ _]\d+\b", re.I), "<feat>"),          # raw SNAP feature names
    (re.compile(r"\b(?:node(?:_id)?)\s*[=: ]\s*\d+\b", re.I), "<node>"),        # node_id=123 / node 123
    (re.compile(r"\b(?:ego(?:_id)?)\s*[=: ]\s*\d+\b", re.I), "<node>"),         # ego_id=107
    (re.compile(r"\b\d{9,}\b"), "<id>"),                                        # long numeric ids
]


def redact(msg: str) -> str:
    for pat, repl in _PATTERNS:
        msg = pat.sub(repl, msg)
    return msg


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"graph.{name}")
    if not logger.handlers:
        log_dir = PROJECT_ROOT / "artifacts" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        for h in (logging.StreamHandler(), logging.FileHandler(log_dir / "encoder.log")):
            h.setFormatter(fmt)
            h.addFilter(RedactingFilter())
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
