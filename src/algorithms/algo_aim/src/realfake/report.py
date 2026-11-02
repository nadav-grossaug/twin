"""Result tables (plan section 2): rows -> csv / json in the layout of ``snap/eval/results``.

One flat row per (method, epsilon, seed, model, ...) with the AUC, its CI and the control numbers
next to it, written as ``<name>.csv`` plus a json mirror, and a ``<name>_summary.csv`` with mean
and std over seeds (same shape as ``snap/eval/score.py``), so the discriminator results sit beside
the existing sweep tables and the notebooks can read both the same way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and std (over seeds) of every numeric column, one row per group (modeled on
    ``snap/eval/score.py``)."""
    num = [c for c in df.columns if c not in keys and c != "seed" and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys, dropna=False, sort=False)
    agg = g[num].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_seeds", g.size())
    return agg.reset_index()


def write_results(
    rows: list[dict], out_dir: str | Path, name: str = "results", summary_keys: list[str] | None = None
) -> pd.DataFrame:
    """Write ``rows`` (flat result-row dicts) as ``<out_dir>/<name>.csv`` and ``<name>.json``;
    with ``summary_keys`` also ``<name>_summary.csv`` (mean/std over seeds per key group).
    Returns the DataFrame. Nothing is written for an empty ``rows``."""
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    df.to_json(out / f"{name}.json", orient="records", indent=2)
    if summary_keys:
        summarize(df, summary_keys).to_csv(out / f"{name}_summary.csv", index=False)
    return df
