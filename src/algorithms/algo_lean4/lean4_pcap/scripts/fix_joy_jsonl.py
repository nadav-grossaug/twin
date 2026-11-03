#!/usr/bin/env python3
"""Repair known Cisco Joy JSON serializer bugs in a flow .jsonl, in place.

Joy sometimes emits an empty element inside the server-certificate
"X509v3 extensions" array -- literally `},,{` -- which makes the whole line
invalid JSON and aborts every consumer (DuckDB read_json, the pytest raw
fixtures, export scripts). The repair is byte-minimal and verified:

  * only lines that FAIL json.loads are touched;
  * the single known fix `},,{` -> `},{` (dropping the empty array element)
    is applied repeatedly, then the line must re-parse or the script exits 1;
  * untouched lines are preserved byte-for-byte; the file is rewritten
    atomically (temp file + rename) only when something was repaired.

Usage: python3 scripts/fix_joy_jsonl.py [flow.jsonl]
Exit codes: 0 ok (repaired or already clean), 1 unrepairable line found.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

KNOWN_BAD, KNOWN_FIX = b"},,{", b"},{"


def repair_line(raw: bytes) -> bytes | None:
    """Return a parseable replacement for a malformed line, or None."""
    fixed = raw
    while KNOWN_BAD in fixed:
        fixed = fixed.replace(KNOWN_BAD, KNOWN_FIX)
    try:
        json.loads(fixed)
    except ValueError:
        return None
    return fixed


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "flow.jsonl")
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    repaired: list[int] = []
    tmp = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    tmp_path = Path(tmp.name)
    try:
        with open(path, "rb") as src, tmp:
            for lineno, raw in enumerate(src, 1):
                stripped = raw.strip()
                if stripped:
                    try:
                        json.loads(stripped)
                    except ValueError:
                        fixed = repair_line(stripped)
                        if fixed is None:
                            print(f"line {lineno}: malformed JSON not covered by the "
                                  "known joy `},,{` repair -- refusing to guess",
                                  file=sys.stderr)
                            return 1
                        raw = raw.replace(stripped, fixed)
                        repaired.append(lineno)
                tmp.write(raw)
        if repaired:
            tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if repaired:
        print(f"{path}: repaired {len(repaired)} line(s): "
              f"{', '.join(map(str, repaired[:10]))}"
              f"{' ...' if len(repaired) > 10 else ''}")
    else:
        print(f"{path}: all lines parse -- nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
