"""Logging that excludes sensitive observations.

Every release-layer logger goes through `RedactingFilter`, which drops
IPv4 addresses, MAC addresses and epoch timestamps from messages. Code
should log counts and stage names only; the filter is the backstop, not
the primary control.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PATTERNS = [
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),                       # IPv4 addresses
    (re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"), "<mac>"),      # aa:bb:cc:dd:ee:ff
    (re.compile(r"\b[0-9A-Fa-f]{12}\b"), "<mac>"),                              # bare 12-hex MACs
    (re.compile(r"\b\d{6,}\.\d+\b"), "<ts>"),                                   # epoch floats (>=6 integer digits)
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
    logger = logging.getLogger(f"flow.{name}")
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
