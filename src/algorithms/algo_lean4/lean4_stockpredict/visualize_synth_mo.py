# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas"]
# ///
"""Synth-vs-real comparison notebook: price paths, per-column TV scorecard
with each column's fill source, planted-pattern recall, collinearity gap,
fidelity report and certificates. Declared-gap framing throughout."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import json
    import math
    from collections import Counter, defaultdict
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parent

    def load_tsv(path):
        out = defaultdict(list)
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                out[r["stockname"]].append(
                    (r["date-yyyy-mm-dd"], float(r["stockprice"])))
        return dict(out)

    real = load_tsv(ROOT / "stock_normalized_train.tsv")
    synth_p = ROOT / "synth" / "stock_synth.tsv"
    synth = load_tsv(synth_p) if synth_p.exists() else {}
    return Counter, ROOT, alt, json, load_tsv, math, mo, pd, real, synth


@app.cell
def _(mo, synth):
    mo.md(
        "# Synthetic vs real — scorecard\n"
        "The generator targets the **spec contract and the DP release**, not "
        "realism; gaps are declared, measured and reported, never hidden."
        + ("" if synth else "\n\n*(no synth yet — run `make synth`)*"))
    return


@app.cell
def _(alt, mo, pd, real, synth):
    _rows = []
    for _src, _d in (("real", real), ("synthetic", synth)):
        for _t, _ps in _d.items():
            for _date, _p in _ps:
                _rows.append({"src": _src, "ticker": _t, "d": _date, "price": _p})
    _df = pd.DataFrame(_rows)
    _chart = alt.Chart(_df).mark_line(opacity=0.8).encode(
        x="d:T", y="price:Q", color="ticker:N").properties(
        width=320, height=240).facet(column="src:N")
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(Counter, alt, math, mo, pd, real, synth):
    def _rets(ps):
        vals = [p for _, p in ps]
        return [round(math.log(b / a) * 10000) for a, b in zip(vals, vals[1:])]

    def _cls(bp):
        if bp <= -300: return "crash"
        if bp < -50: return "down"
        if bp <= 50: return "flat"
        if bp < 300: return "up"
        return "surge"

    def _mag(bp):
        a = abs(bp)
        if a <= 50: return "mag_0"
        if a <= 150: return "mag_1"
        if a < 300: return "mag_2"
        if a < 600: return "mag_3"
        if a < 1200: return "mag_4"
        return "mag_5"

    def _dow(i):
        return ["mon", "tue", "wed", "thu", "fri"][i % 5]

    def _col(d, fn, use_idx=False):
        c = Counter()
        for ps in d.values():
            for i, bp in enumerate(_rets(ps)):
                c[fn(i + 1) if use_idx else fn(bp)] += 1
        return c

    def _tv(a, b):
        ta, tb = sum(a.values()), sum(b.values())
        keys = set(a) | set(b)
        return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in keys) if ta and tb else None

    _score = []
    if synth:
        for _name, _fn, _ui, _srcd in [
            ("return class", _cls, False, "DP class mix + transitions + priors"),
            ("magnitude bin", _mag, False, "DP mag_bins ∩ class bins + vol boost"),
            ("day of week", _dow, True, "public calendar (structural)"),
        ]:
            _a, _b = _col(real, _fn, _ui), _col(synth, _fn, _ui)
            _score.append({"column": _name, "tv_real_vs_synth": round(_tv(_a, _b), 4),
                           "fill_source": _srcd})
    _df = pd.DataFrame(_score)
    _out = mo.md("*(run `make synth`)*")
    if len(_df):
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("tv_real_vs_synth:Q", scale=alt.Scale(domain=[0, 1])),
            y="column:N", color=alt.Color("fill_source:N"),
            tooltip=["column", "tv_real_vs_synth", "fill_source"]).properties(
            width=520, height=140, title="Per-column TV, real vs synth")
        _out = mo.vstack([mo.md("## TV scorecard"), mo.ui.altair_chart(_chart),
                          mo.ui.table(_df, selection=None)])
    _out
    return


@app.cell
def _(ROOT, json, mo):
    _items = []
    for _name, _path in [
        ("validation", ROOT / "synth" / "validation.json"),
        ("certificate", ROOT / "synth" / "certificate.json"),
        ("distill fidelity", ROOT / "artifacts" / "reports" / "distill_report.json"),
        ("realism audit", ROOT / "artifacts" / "reports" / "realism_audit.json"),
    ]:
        if _path.exists():
            _items.append(mo.vstack([
                mo.md(f"**{_name}** (`{_path.name}`)"),
                mo.json(json.loads(_path.read_text())),
            ]))
    mo.vstack([mo.md("## Contracts, certificates and honest reports")] + _items)
    return


@app.cell
def _(ROOT, json, mo):
    _p = ROOT / "artifacts" / "reports" / "realism_audit.json"
    _out = mo.md("")
    if _p.exists():
        _r = json.loads(_p.read_text())
        _c = _r.get("collinearity", {})
        _out = mo.md(
            f"## The declared collinearity gap, quantified\n"
            f"Real mean same-sign share "
            f"**{_c.get('real', {}).get('mean_same_sign_share')}** vs synth "
            f"**{_c.get('synth', {}).get('mean_same_sign_share')}**; real mean "
            f"Pearson r **{_c.get('real', {}).get('mean_pearson_r')}** vs synth "
            f"**{_c.get('synth', {}).get('mean_pearson_r')}**. The market "
            f"coupling (λ from the DP co-movement release) transfers sign "
            f"agreement, not the full correlation matrix — a declared "
            f"`fields_not_modelled` gap, not a silent one.")
    _out
    return


if __name__ == "__main__":
    app.run()
