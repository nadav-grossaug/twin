"""STAGE 3 (eval): synthetic traces (+ real traces) -> method-M metrics and the real-vs-synthetic comparison.

For every variant written by stage 2 (`geolife/outside/synthetic/index.json`):
  * rebuild the public map from the variant's meta.json,
  * run method M v2 (places of interest, common places / segments, co-visit relations, communities) on the synthetic
    traces and score it against the PLANTED truth stored next to them,
  * if real points and the public places table are given: run the same M on the real traces and compare the
    distribution statistics (outings per user, stops per outing, users per place, ...) with the synthetic ones.
Results go to `--out` (default geolife/eval/results): `<variant>.json` per variant and `summary.csv`.
Eval is an inside-only zone: it reads real traces. Never copy its results across the boundary without review.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from geolife.eval.m.core import MapView
from geolife.eval.m.outings import M2Config, distribution_stats, run_m2
from geolife.eval.m.outings_transfer import _clean, compare_stats, evaluate, make_mv, real_outings, row
from geolife.outside.providers import make_provider
from geolife.outside.synthetic_io import read_synthetic


def score_variant(syn_dir, cfg: M2Config | None = None, real_points=None, places=None) -> dict:
    cfg = cfg or M2Config()
    g = read_synthetic(syn_dir)
    spec = g.meta["map"]
    provider = make_provider(spec)
    mv = make_mv(provider, with_graph=spec["kind"] == "grid")
    res = run_m2(g.result.traces, mv, cfg)
    out = {"variant": g.meta["variant"], "artifact": g.meta["artifact"], "n_users": g.meta["n_users"],
           "epsilon": g.meta.get("epsilon"), "eval": evaluate(g, res), "stats": distribution_stats(res)}
    if real_points and places:
        uos = real_outings(cfg, places, points_path=str(real_points))
        if uos:
            from geolife.eval.m.core import MapView as _MV
            from geolife.eval.m.outings import analyze, place_merge_map
            from geolife.transfer.tokenize.stays import PlaceTable
            mvr = _MV(PlaceTable.from_parquet(places), None)
            rres = analyze(uos, cfg, has_graph=False, merge=place_merge_map(mvr, cfg.merge_m))
            out["real_stats"] = distribution_stats(rres)
            out["compare"] = compare_stats(out["stats"], out["real_stats"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", default="geolife/outside/synthetic")
    ap.add_argument("--out", default="geolife/eval/results")
    ap.add_argument("--only", default=None, help="only variants starting with this prefix")
    ap.add_argument("--real-points", default=None, help="real points.parquet (uid, t, lat, lon) for the comparison")
    ap.add_argument("--places", default=None, help="public places parquet, with --real-points")
    a = ap.parse_args(argv)
    syn, out = Path(a.synthetic), Path(a.out)
    index = json.loads((syn / "index.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ent in index:
        if a.only and not ent["dir"].startswith(a.only):
            continue
        r = score_variant(syn / ent["dir"], real_points=a.real_points, places=a.places)
        (out / f"{ent['dir']}.json").write_text(json.dumps(_clean(r), indent=1) + "\n", encoding="utf-8")
        rows.append({"dir": ent["dir"], "epsilon": r["epsilon"], **row(r["eval"])})
        print(ent["dir"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()},
              file=sys.stderr)
    if rows:
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
