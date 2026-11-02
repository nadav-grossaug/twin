"""STAGE 3 (eval) helper: attribute coupling of synthetic graphs against the real graph, as a table and a CSV.

Usage (repo root):
    PYTHONPATH=src python snap/eval/coupling_report.py --real PATH --syn PATH [PATH ...] [--out CSV]
        [--policy snap/inside/attribute_policy.json] [--bin-edges 0 1 3 ...] [--exclude-hubs 10]

Prints one compact table per synthetic graph (lift, assortativity, mixing TVD, degree-by-value TVD, eta2, and the gap
closed against the permuted-attribute null) and writes a long-form CSV (graph, attribute, metric, value). The metrics
are in ``graph_aim.eval.coupling``; ``missing_label`` / ``ordered`` per attribute come from the policy file (optional).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from graph_aim.eval.coupling import coupling_metrics, gap_closed_table, options_from_policy, permuted_null
from graph_aim.eval.evaluate import align_columns, degrees
from graph_aim.inside.measure import degree_bins
from graph_aim.transfer.graph_io import load_attributed_graph

SNAP = Path(__file__).resolve().parent.parent
POLICY = SNAP / "inside" / "attribute_policy.json"


def build_table(m: dict, gaps: dict) -> pd.DataFrame:
    rows = []
    for col, a in m["attributes"].items():
        g = gaps["attributes"][col]
        rows.append(
            {
                "attribute": col,
                "lift_real": a["lift_real"],
                "lift_syn": a["lift_syn"],
                "assort_real": a["assort_real"],
                "assort_syn": a["assort_syn"],
                "mix_tvd": a["mix_tvd"],
                "deg_tvd": a["deg_by_value_tvd"],
                "eta2_real": a["eta2_real"],
                "eta2_syn": a["eta2_syn"],
                "gap_lift": g.get("lift"),
                "gap_eta2": g.get("eta2"),
                "gap_deg_tvd": g.get("deg_by_value_tvd"),
            }
        )
    return pd.DataFrame(rows)


def long_form(name: str, m: dict, gaps: dict | None = None) -> pd.DataFrame:
    rows = [(name, col, k, v) for col, a in m["attributes"].items() for k, v in a.items()]
    rows += [(name, "mean", k, v) for k, v in m["mean"].items()]
    if gaps is not None:
        rows += [(name, col, f"gap_{k}", v) for col, g in gaps["attributes"].items() for k, v in g.items()]
        rows += [(name, "mean", f"gap_{k}", v) for k, v in gaps["mean"].items()]
    return pd.DataFrame(rows, columns=["graph", "attribute", "metric", "value"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--syn", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="long-form CSV")
    ap.add_argument("--policy", default=str(POLICY), help="attribute policy (optional keys missing_label, ordered)")
    ap.add_argument("--bin-edges", nargs="+", type=int, default=None, help="public degree bin edges")
    ap.add_argument("--exclude-hubs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="seed of the permuted-attribute null")
    args = ap.parse_args()

    real = load_attributed_graph(args.real)
    missing, ordered = {}, {}
    if args.policy and Path(args.policy).exists():
        missing, ordered = options_from_policy(json.loads(Path(args.policy).read_text()))
    dmax = int(degrees(real).max()) if real.n else 0
    edges = args.bin_edges if args.bin_edges else degree_bins(dmax, 6, dmax + 1)
    kw = dict(degree_bin_edges=edges, missing=missing, ordered=ordered, exclude_hubs=args.exclude_hubs)
    print(f"degree bin edges {list(edges)}; hubs excluded for the _nohub metrics: {args.exclude_hubs}")

    m_null = coupling_metrics(real, permuted_null(real, args.seed), **kw)
    frames = [long_form("permuted_null", m_null)]
    for path in args.syn:
        name = Path(path).stem
        syn = align_columns(load_attributed_graph(path), real)
        m = coupling_metrics(real, syn, **kw)
        gaps = gap_closed_table(m_null, m)
        print(f"\n== {name} ==")
        print(build_table(m, gaps).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        frames.append(long_form(name, m, gaps))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
