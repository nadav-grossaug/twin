"""python -m geolife.transfer.artifact_cli {verify,inspect,sign} <dir> [--key KEYFILE]

verify  : hashes + leakage/ledger validator; with --key also the HMAC signature.
inspect : human-readable report of the artifact.
sign    : write the HMAC-SHA256 signature over manifest.json (shared-key authenticity; the key
          file lives INSIDE the air gap and travels to the verifier over the trusted code channel).
"""
from __future__ import annotations

import argparse
import sys

from geolife.transfer.artifact import (ArtifactError, load_artifact, sign_artifact,
                                       validate_no_raw_leakage, verify_signature)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="geolife.transfer.artifact_cli")
    ap.add_argument("cmd", choices=["verify", "inspect", "sign"])
    ap.add_argument("dir")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--user-id-regex", default=None)
    ap.add_argument("--key", default=None, help="HMAC key file (sign: required; verify: optional)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "sign":
            if not a.key:
                ap.error("sign requires --key")
            load_artifact(a.dir)                     # never sign a broken artifact
            p = sign_artifact(a.dir, a.key)
            print(f"OK: wrote {p}")
            return 0
        art = load_artifact(a.dir)
        kw = {"user_id_regex": a.user_id_regex} if a.user_id_regex else {}
        if a.cmd == "verify":
            validate_no_raw_leakage(art, **kw)
            if a.key:
                verify_signature(a.dir, a.key)
            print(f"OK: {len(art.manifest.files)} files verified, no leakage flags"
                  + (", signature valid" if a.key else ""))
            return 0
        from geolife.transfer.inspect import render_report
        print(render_report(art, markdown=a.markdown))
        return 0
    except ArtifactError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
