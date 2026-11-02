"""Throwaway: digest the lever-sweep coupling CSVs into pass/fail tables vs the current config.

Run (repo root): python3 tools/sweep/digest.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
IN_SCOPE = ["school", "birthday", "hometown", "location", "locale", "employer"]
VARIANTS = {
    "current (0.20 / 6 bins)": ROOT / "tools" / "coupling_after.csv",
    "share35 (0.35 / 6 bins)": ROOT / "tools" / "sweep" / "share35" / "coupling.csv",
    "bins4 (0.20 / 4 bins)": ROOT / "tools" / "sweep" / "bins4" / "coupling.csv",
    "share35_bins4 (0.35 / 4 bins)": ROOT / "tools" / "sweep" / "share35_bins4" / "coupling.csv",
}


def table(path, eps_variant):
    df = pd.read_csv(path)
    df = df[df["graph"].str.startswith(f"{eps_variant}_seed")]
    p = df.pivot_table(index="attribute", columns="metric", values="value", aggfunc="mean")
    s = df.pivot_table(index="attribute", columns="metric", values="value", aggfunc="std")
    return p, s


def main() -> None:
    for eps_variant, thr in [("graph_aim_eps4", 0.3), ("graph_aim_epsinf", 0.5)]:
        print(f"\n## {eps_variant} (threshold >= {thr} on gap_eta2 AND gap_deg_by_value_tvd, 3-seed mean)\n")
        print("| variant | " + " | ".join(IN_SCOPE) + " | pass |")
        print("|" + "---|" * (len(IN_SCOPE) + 2))
        for name, path in VARIANTS.items():
            if not path.exists():
                continue
            p, s = table(path, eps_variant)
            cells, npass = [], 0
            for a in IN_SCOPE:
                ge, gd = p.loc[a, "gap_eta2"], p.loc[a, "gap_deg_by_value_tvd"]
                ok = (ge >= thr) and (gd >= thr)
                npass += ok
                cells.append(f"{ge:.2f}/{gd:.2f}{' *' if ok else ''}")
            print(f"| {name} | " + " | ".join(cells) + f" | {npass}/6 |")
        print("\n(cell = gap_eta2 / gap_deg_tvd; * = passes both)")


if __name__ == "__main__":
    main()
