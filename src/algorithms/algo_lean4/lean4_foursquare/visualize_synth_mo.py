# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas"]
# ///
"""Synth-vs-real comparison notebook: travel-count tail, region mix, gap
classes, corridor structure, per-column TV scorecard with each column's fill
source, planted-pattern recall and certificates. Declared-gap framing
throughout."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import json
    from collections import Counter, defaultdict
    from datetime import datetime
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(ROOT / "python"))
    from spec import REGIONS, load_spec, region_of  # noqa: E402

    def load_tsv(path):
        out = defaultdict(list)
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                out[r["userid"]].append(r)
        for rows in out.values():
            rows.sort(key=lambda r: r["dt"])
        return dict(out)

    real = load_tsv(ROOT / "country_transfer.tsv")
    synth_p = ROOT / "synth" / "travel_synth.tsv"
    synth = load_tsv(synth_p) if synth_p.exists() else {}
    return (Counter, ROOT, alt, datetime, json, load_spec, mo, pd, real,
            region_of, synth)


@app.cell
def _(mo, synth):
    mo.md(
        "# Synthetic vs real travel — scorecard\n"
        "The generator targets the **spec contract and the DP release**, not "
        "realism; gaps are declared, measured and reported, never hidden."
        + ("" if synth else "\n\n*(no synth yet — run `make synth`)*"))
    return


@app.cell
def _(Counter, alt, mo, pd, real, synth):
    def _tc_bin(n):
        if n == 1: return "tc_1"
        if n == 2: return "tc_2"
        if n <= 4: return "tc_3_4"
        if n <= 8: return "tc_5_8"
        if n <= 16: return "tc_9_16"
        if n <= 32: return "tc_17_32"
        return "tc_33p"

    _rows = []
    for _src, _d in (("real", real), ("synthetic", synth)):
        _c = Counter(_tc_bin(len(rs)) for rs in _d.values())
        _tot = max(1, sum(_c.values()))
        for _b, _n in _c.items():
            _rows.append({"src": _src, "bin": _b, "share": _n / _tot})
    _order = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16", "tc_17_32", "tc_33p"]
    _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
        x=alt.X("bin:N", sort=_order), y="share:Q",
        color="src:N", xOffset="src:N").properties(
        width=480, height=220,
        title="Travels-per-user (the migration's first focus): real vs synth")
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(Counter, alt, datetime, mo, pd, real, region_of, synth):
    def _gap_class(s):
        if s < 172800: return "gap_hop"
        if s < 604800: return "gap_trip"
        if s < 2592000: return "gap_monthly"
        if s < 7776000: return "gap_quarterly"
        return "gap_rare"

    def _ts(r):
        return datetime.strptime(r["dt"], "%Y-%m-%d %H:%M:%S").timestamp()

    def _cols(d):
        reg, gap, dur = Counter(), Counter(), Counter()
        for rows in d.values():
            ts = [_ts(r) for r in rows]
            for r in rows:
                reg[region_of(r["next_country"])] += 1
                nd = int(r["next_duration"])
                dur["dur_day" if nd < 86400 else
                    "dur_week" if nd < 604800 else
                    "dur_month" if nd < 2592000 else
                    "dur_quarter" if nd < 7776000 else "dur_long"] += 1
            for a, b in zip(ts, ts[1:]):
                gap[_gap_class(b - a)] += 1
        return {"next region": reg, "gap class": gap, "duration class": dur}

    def _tv(a, b):
        ta, tb = sum(a.values()), sum(b.values())
        keys = set(a) | set(b)
        return (0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in keys)
                if ta and tb else None)

    _score = []
    if synth:
        _ra, _sa = _cols(real), _cols(synth)
        _fill = {
            "next region": "DP region mix + 8x8 transitions + return prior",
            "gap class": "DP gap-class histogram + suppressed moments",
            "duration class": "DP duration-class histogram",
        }
        for _name in _ra:
            _t = _tv(_ra[_name], _sa[_name])
            _score.append({"column": _name,
                           "tv_real_vs_synth": None if _t is None else round(_t, 4),
                           "fill_source": _fill[_name]})
    _df = pd.DataFrame(_score)
    _out = mo.md("*(run `make synth`)*")
    if len(_df):
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("tv_real_vs_synth:Q", scale=alt.Scale(domain=[0, 1])),
            y="column:N", color=alt.Color("fill_source:N"),
            tooltip=["column", "tv_real_vs_synth", "fill_source"]).properties(
            width=520, height=140, title="Per-column TV, real vs synth")
        _out = mo.vstack([mo.md(
            "## TV scorecard\n"
            "**TV = total variation distance** between the real and synthetic "
            "distributions over a column's public bins: "
            "TV = ½ · Σ|p_real(bin) − p_synth(bin)|, ranging **0 to 1** — "
            "0 means identical mixes, 1 means completely disjoint. Read it as "
            "*the share of probability mass that would have to move to make "
            "the two distributions match* (TV 0.05 ⇒ 5% of mass misplaced). "
            "The same statistic gates `make synth` validation (soft tolerance "
            "0.25) — the generator targets the DP release, so some drift from "
            "the *real* data is expected and honest."),
            mo.ui.altair_chart(_chart),
            mo.ui.table(_df, selection=None)])
    _out
    return


@app.cell
def _(Counter, mo, pd, real, synth):
    _rows = []
    for _src, _d in (("real", real), ("synthetic", synth)):
        _c = Counter()
        for _rs in _d.values():
            for _r in _rs:
                _c[(_r["prev_country"], _r["next_country"])] += 1
        for (_a, _b), _n in _c.most_common(8):
            _rows.append({"src": _src, "corridor": f"{_a}→{_b}", "n": _n})
    mo.vstack([
        mo.md("## Top corridors (sequence structure)\n"
              "Synth corridors come ONLY from the released top-k list "
              "(suppression ≥50 users / ≥100 transfers); rare real corridors "
              "cannot leak because they were never released."),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ])
    return


@app.cell
def _(ROOT, json, mo):
    _items = []
    for _name, _path in [
        ("validation", ROOT / "synth" / "validation.json"),
        ("certificate", ROOT / "synth" / "certificate.json"),
        ("distill fidelity", ROOT / "models_out" / "fidelity.json"),
        ("realism audit", ROOT / "artifacts" / "reports" / "realism_audit.json"),
        ("synth meta (assumptions)", ROOT / "synth" / "meta.json"),
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
    _p = ROOT / "synth" / "meta.json"
    _out = mo.md("")
    if _p.exists():
        _gaps = json.loads(_p.read_text())["assumptions"]["fields_not_modelled"]
        _out = mo.md(
            "## Declared gaps (fields_not_modelled)\n" +
            "\n".join(f"- {g}" for g in _gaps) +
            "\n\nDeclared gaps over silent fabrication: each is measured by "
            "the realism audit, none is hidden behind plausible noise.")
    _out
    return


if __name__ == "__main__":
    app.run()
