"""python -m geolife.transfer.artifact_cli {verify,inspect} <dir>"""
from __future__ import annotations

import argparse
import sys

from geolife.transfer.artifact import ArtifactError, load_artifact, validate_no_raw_leakage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="geolife.transfer.artifact_cli")
    ap.add_argument("cmd", choices=["verify", "inspect"])
    ap.add_argument("dir")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--user-id-regex", default=None)
    a = ap.parse_args(argv)
    try:
        art = load_artifact(a.dir)
        kw = {"user_id_regex": a.user_id_regex} if a.user_id_regex else {}
        if a.cmd == "verify":
            validate_no_raw_leakage(art, **kw)
            print(f"OK: {len(art.manifest.files)} files verified, no leakage flags")
            return 0
        from geolife.transfer.inspect import render_report
        print(render_report(art, markdown=a.markdown))
        return 0
    except ArtifactError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
