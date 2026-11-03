"""STAGE 1 (inside): real GeoLife traces -> DP artifacts in geolife/transfer/artifacts/<variant>_seed<k>/.

    PYTHONPATH=src python geolife/inside/fit_artifacts.py --points geolife/inside/data/points.parquet \
        --places geolife/outside/cache/places_beijing_big.parquet [--variant outings_v2] [--seeds 3] [--epsilon 6]

Reads the real traces (points.parquet: uid, t, lat, lon, built from the raw dataset in geolife/inside/data/, see
geolife/inside/data/README.md) and the PUBLIC places table, tokenizes outings (pseudonymised), caps each user's
contribution, adds Gaussian noise under user-level zCDP (eps = 6, delta = 1e-6 by default) and writes one artifact per
seed. Nothing else leaves: --report writes a LOCAL signal-to-noise file (never copy it across). Loading and
validating each artifact (hashes, privacy ledger, provenance allowlist, leakage guard) is part of the run.

Release hardening (PLAN step 7):
  --release    one-seed-per-release policy: exactly one seed is produced (--seeds must be 1 or
               absent) and the policy is recorded in privacy.json (seeds of one fit are noise
               replicates; averaging k of them cuts the noise std by sqrt(k)). Dev runs may still
               use several seeds, but only one may ever leave the air gap.
  --sign-key F sign each artifact: HMAC-SHA256 over manifest.json with the shared key in file F
               (generate with e.g. `python -c "import secrets;print(secrets.token_hex(32))" > F`).
               Shared-key authenticity, not a public-key signature; keep F inside the air gap and
               hand it to the outside verifier over the trusted channel that moves the code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geolife.inside.data import default_data_dir
from geolife.inside.learn_v2 import (BEIJING_CORE, ONE_SEED_CAVEAT, _clean, multi_seed_caveat,
                                     run, users_from_points)
from geolife.inside.measure_sensors import sensor_views_from_points
from geolife.transfer.artifact import sign_artifact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default=None, help="points.parquet (default: <data dir>/points.parquet)")
    ap.add_argument("--places", required=True, help="public places parquet (id, category, lat, lon, popularity)")
    ap.add_argument("--data", default=None, help="GeoLife root holding Data/<uid>/labels.txt (default: data dir)")
    ap.add_argument("--variant", default="outings_v2")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=6.0)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--bbox", type=float, nargs=4, default=list(BEIJING_CORE))
    ap.add_argument("--out", default="geolife/transfer/artifacts")
    ap.add_argument("--report-dir", default=None, help="local SNR json per seed, e.g. geolife/inside/results")
    ap.add_argument("--no-sensors", action="store_true", help="skip the sensor tables (pre-D20 behaviour)")
    ap.add_argument("--release", action="store_true",
                    help="one-seed release policy: forces --seeds 1 and stamps the policy into privacy.json")
    ap.add_argument("--sign-key", default=None, help="HMAC key file; signs every artifact written")
    a = ap.parse_args(argv)
    if a.release and a.seeds != 1:
        ap.error("--release publishes exactly one seed; drop --seeds or set it to 1")

    import pandas as pd
    root = Path(a.data) if a.data else default_data_dir()
    points = Path(a.points) if a.points else next(
        (c / "points.parquet" for c in (root, root.parent) if (c / "points.parquet").exists()), root / "points.parquet")
    cats = sorted(set(pd.read_parquet(a.places, columns=["category"]).category.astype(str)))
    users = users_from_points(points, a.places, root, seed=0)
    print(f"{len(users)} users with outings", file=sys.stderr)
    sensor_views = None
    if not a.no_sensors:
        sensor_views = sensor_views_from_points(points, root, seed=0, bbox=a.bbox)
        print(f"{len(sensor_views)} sensor views", file=sys.stderr)
    out = Path(a.out)
    index = []
    for k in range(a.seeds):
        name = f"{a.variant}_seed{k}"
        extra = ((ONE_SEED_CAVEAT,) if a.release
                 else (multi_seed_caveat(a.seeds),) if a.seeds > 1 else ())
        res = run(users, out / name, epsilon=a.epsilon, delta=a.delta, bbox=a.bbox, categories=cats, seed=k,
                  with_snr=bool(a.report_dir), sensor_views=sensor_views, extra_caveats=extra)
        if a.sign_key:
            sign_artifact(out / name, a.sign_key)
            print(f"signed {name} (HMAC-SHA256 over manifest.json)", file=sys.stderr)
        led = res["ledger"]
        index.append({"variant": a.variant, "epsilon": led.epsilon, "delta": led.delta, "seed": k, "dir": name,
                      **({"release": True} if a.release else {}), **({"signed": True} if a.sign_key else {})})
        if a.report_dir:
            rd = Path(a.report_dir)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / f"{name}_snr.json").write_text(json.dumps(_clean({"snr": res["snr"], "epsilon": led.epsilon,
                                                                  "rho": res["rho"]}), indent=1))
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
