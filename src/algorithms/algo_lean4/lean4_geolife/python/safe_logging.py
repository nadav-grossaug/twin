"""Logging that excludes sensitive observations.

Every release-layer logger goes through `RedactingFilter`, which drops
user ids, coordinates, trajectory/sequence ids and epoch timestamps from
messages. Code should log counts and stage names only; the filter is the
backstop, not the primary control.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PATTERNS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                re.I), "<seq>"),                                                 # uuid-shaped sequence ids
    (re.compile(r"\b(?:user(?:_?id)?)\s*[=: ]\s*\d+\b", re.I), "<user>"),        # user_id=000 / user 181
    (re.compile(r"\b\d{9,}\b"), "<ts>"),                                         # epoch seconds, yyyymmddhhmmss trajectory ids
    (re.compile(r"\b\d{6,}\.\d+\b"), "<ts>"),                                    # epoch floats (>=6 integer digits)
    (re.compile(r"[-+]?\b\d{1,3}\.\d{3,}\b"), "<coord>"),                        # lat/lon decimal degrees
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
    logger = logging.getLogger(f"geo.{name}")
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
