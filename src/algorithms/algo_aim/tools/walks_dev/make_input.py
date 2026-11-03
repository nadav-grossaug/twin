"""Throwaway (eval zone, reads the real graph): build the real-scale W2 test input.

The exact base artifact (v2 config, noise off) + the walk tables of the real SNAP graph (200,000 walks), wrapped in
a ``SimpleNamespace`` with the shape of W1's ``WalksArtifact`` (``.base``, ``.spec``, ``.tables``, ``.n_walks``),
pickled to the path given (default: ``tools/walks_dev/cache/input.pkl``, not committed).

Usage (repo root): ``PYTHONPATH=src /opt/venv/bin/python3 tools/walks_dev/make_input.py [out.pkl]``
"""

import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from graph_aim.inside.fit import fit
from graph_aim.transfer.graph_io import load_attributed_graph
from graph_aim_walks.transfer import walkstats as ws

REPO = Path(__file__).resolve().parents[2]
N_WALKS = 200_000

SNAP_SPEC = ws.WalkSpec(
    groups=(("school",), ("hometown",), ("location",), ("hometown", "location")),
    share_attrs=("gender", "locale", "birthday", "school", "employer", "location", "hometown", "language"),
    dominance_pairs=(("hometown", "location"), ("hometown", "school"), ("school", "locale")),
)


def build(out: Path) -> None:
    t0 = time.time()
    g = load_attributed_graph(REPO / "snap" / "inside" / "data" / "processed" / "attributed_graph.npz")
    cfg = json.loads((REPO / "snap" / "inside" / "config" / "graph_aim_v2.json").read_text())
    cfg["noise"] = False
    base = fit(g, cfg, seed=0)
    print(f"base fitted ({time.time() - t0:.0f}s)", flush=True)
    attrs = {c: g.attrs[c].to_numpy() for c in g.attrs.columns}
    tables = ws.walk_tables(g.edges, g.n, attrs, g.domains, SNAP_SPEC, N_WALKS, seed=0)
    tables["transition_spectrum"] = ws.transition_spectrum(g.edges, g.n)
    print(f"tables ({time.time() - t0:.0f}s)", flush=True)
    art = SimpleNamespace(base=base, spec=SNAP_SPEC, tables=tables, n_walks=N_WALKS)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(art))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "tools" / "walks_dev" / "cache" / "input.pkl")
