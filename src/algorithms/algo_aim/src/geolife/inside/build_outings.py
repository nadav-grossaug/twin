"""Real points.parquet + public places -> outings parquet (stage-1 helper, inside only).

    PYTHONPATH=src python -m geolife.inside.build_outings --points geolife/inside/data/points.parquet \
        --places geolife/outside/cache/places_beijing.parquet --out geolife/inside/results/outings_v0.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geolife.inside.data import default_data_dir, read_labels
from geolife.inside.loaders import load_points_arrays
from geolife.transfer.geo import pseudonyms
from geolife.transfer.tokenize.outings import OutingsConfig, PlaceIndex, to_frames, user_outings_from_arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None, help="GeoLife root (labels.txt) ")
    ap.add_argument("--points", default=None, help="points.parquet cache (uid, traj, t, lat, lon)")
    ap.add_argument("--places", required=True)
    ap.add_argument("--out", default="geolife/inside/results/outings_v0.parquet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(a.data) if a.data else default_data_dir()
    if a.points:
        pts = load_points_arrays(Path(a.points))
    else:
        raise SystemExit("pass --points (parquet cache of uid, traj, t, lat, lon)")
    places = pd.read_parquet(a.places)
    cfg = OutingsConfig()
    pidx = PlaceIndex(places)
    names = pseudonyms(list(pts), a.seed)
    data_dir = root / "Data"
    stats: dict = {}
    users = []
    for uid in sorted(pts):
        lp = data_dir / uid / "labels.txt"
        labs = read_labels(lp) if lp.exists() else []
        uo = user_outings_from_arrays(names[uid], *pts[uid], pidx, cfg, labs, stats)
        if uo is not None:
            users.append(uo)
    of, vf = to_frames(users)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    of.to_parquet(out)
    vf.to_parquet(out.with_name(out.stem + "_visits.parquet"))
    n = len(of)
    st = (of.n_stops.sum() / n) if n else float("nan")
    print(f"users in={len(pts)} kept={len(users)} outings(all, pre-filter users)={stats.get('outings_all')} "
          f"real(kept users)={n} stops={int(of.n_stops.sum())} stops/outing={st:.2f}")
    print("mode:", of["mode"].value_counts().to_dict())
    if len(vf):
        print("snapped share (start/end/stop):",
              vf.assign(s=~vf.place_id.str.startswith("h3:")).groupby("kind").s.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
