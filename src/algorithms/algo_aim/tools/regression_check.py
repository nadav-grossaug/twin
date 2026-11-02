"""Throwaway: compare the new snap/eval/results/sweep_summary.csv structure metrics against the committed
baseline (pass the baseline CSV extracted with `git show` as argv[1]). Flags a metric whose new mean moved
away from the baseline mean by more than the seed-to-seed spread (max of old and new std).

Run: python3 tools/regression_check.py /tmp/baseline_sweep_summary.csv
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = [
    "degree_ks", "degree_emd", "triangles_rel", "clustering_abs", "transitivity_abs",
    "assortativity_abs", "attr_tvd_2way_mean", "attr_tvd_1way_mean", "homophily_error",
    "utility_auc_tstr",
]


def main() -> None:
    old = pd.read_csv(sys.argv[1]).set_index(["method", "epsilon"])
    new = pd.read_csv(ROOT / "snap" / "eval" / "results" / "sweep_summary.csv").set_index(["method", "epsilon"])
    rows = []
    for key in old.index:
        if key not in new.index:
            continue
        o, n = old.loc[key], new.loc[key]
        for m in METRICS:
            om, osd, nm, nsd = o[f"{m}_mean"], o[f"{m}_std"], n[f"{m}_mean"], n[f"{m}_std"]
            spread = max(osd, nsd)
            moved = abs(nm - om) > spread
            rows.append({"method": key[0], "epsilon": key[1], "metric": m,
                         "old": om, "old_std": osd, "new": nm, "new_std": nsd,
                         "delta": nm - om, "spread": spread, "flag": "MOVED" if moved else ""})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFlagged rows only:")
    print(df[df["flag"] == "MOVED"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(ROOT / "tools" / "regression_check.csv", index=False)


if __name__ == "__main__":
    main()
