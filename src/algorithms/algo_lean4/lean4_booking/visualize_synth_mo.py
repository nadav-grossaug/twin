# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb"]
# ///
"""Synth-vs-real comparison notebook: reservations-per-user mix, country
tiers, stay nights, inter-leg gaps, seasonality, per-column TV scorecard
with each column's fill source, top city corridors, planted-pattern recall
and certificates. Declared-gap framing throughout."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parent

    # the public tier grid from the dbt SEED (never inline: a spec bump must
    # not leave the notebook drifting)
    CT_CELLS = []
    for _l in (ROOT / "dbt" / "seeds" / "country_tiers.csv").read_text(
            ).splitlines()[1:]:
        if _l.strip():
            CT_CELLS.append(_l.split(",")[1].strip())
    CT_CELLS.append("ct_other")

    def ct_of(c):
        return c if c in CT_CELLS else "ct_other"

    def rc_bin(n):
        if n <= 3: return "rc_1_3"
        if n == 4: return "rc_4"
        if n == 5: return "rc_5"
        if n == 6: return "rc_6"
        if n <= 8: return "rc_7_8"
        if n <= 12: return "rc_9_12"
        return "rc_13p"

    def night_cell(n):
        if n == 1: return "n_1"
        if n == 2: return "n_2"
        if n == 3: return "n_3"
        if n <= 6: return "n_4_6"
        return "n_7p"

    def gap_cell(g):
        if g == 0: return "g_0"
        if g == 1: return "g_1"
        if g <= 3: return "g_2_3"
        return "g_4p"

    def mixes(con, path):
        """The public-bin mixes of one CSV (real or synthetic)."""
        src = f"read_csv_auto('{path}')"
        res = dict(con.execute(
            f"select user_id, count(*) n from {src} group by 1").fetchall())
        rc = Counter(rc_bin(n) for n in res.values())
        legs = dict(con.execute(
            f"select l, count(*) from (select count(*) l from "
            f"{src} group by utrip_id) group by 1").fetchall())
        nights = Counter(con.execute(
            f"select night_cell, count(*) from (select date_diff('day', "
            f"checkin, checkout) nights, case when date_diff('day', checkin, "
            f"checkout) = 1 then 'n_1' when date_diff('day', checkin, checkout)"
            f" = 2 then 'n_2' when date_diff('day', checkin, checkout) = 3 then "
            f"'n_3' when date_diff('day', checkin, checkout) <= 6 then 'n_4_6' "
            f"else 'n_7p' end night_cell from {src}) group by 1").fetchall())
        gaps = Counter(con.execute(
            f"select case when gap_days = 0 then 'g_0' when gap_days = 1 then "
            f"'g_1' when gap_days <= 3 then 'g_2_3' else 'g_4p' end, count(*) "
            f"from (select date_diff('day', lag(checkout) over (partition by "
            f"utrip_id order by checkin), checkin) gap_days from {src}) "
            f"where gap_days is not null group by 1").fetchall())
        ct = Counter(con.execute(
            f"select case when hotel_country in ({','.join(repr(c) for c in CT_CELLS[:-1])}) "
            f"then hotel_country else 'ct_other' end, count(*) from {src} "
            f"group by 1").fetchall())
        mon = Counter(con.execute(
            f"select strftime(checkin, '%m'), count(*) from {src} group by 1"
        ).fetchall())
        mon_abs = Counter(con.execute(
            f"select strftime(checkin, '%Y-%m'), count(*) from {src} "
            f"group by 1").fetchall())
        dev = Counter(con.execute(
            f"select device_class, count(*) from {src} group by 1").fetchall())
        rep = con.execute(
            f"select count(*) filter (where dc < legs), count(*) from ("
            f"select utrip_id, count(distinct city_id) dc, count(*) legs "
            f"from {src} group by 1)").fetchone()
        cons = con.execute(
            f"select count(*) filter (where prev = city_id), count(*) from ("
            f"select city_id, lag(city_id) over (partition by utrip_id order "
            f"by checkin) prev from {src}) where prev is not null").fetchone()
        tpu = con.execute(
            f"select count(*), count(distinct user_id) from ("
            f"select user_id, utrip_id from {src} group by 1, 2)").fetchone()
        devc = con.execute(
            f"select count(*) filter (where n_dc = 1), count(*) from ("
            f"select utrip_id, count(distinct device_class) n_dc from {src} "
            f"group by 1)").fetchone()
        affs = con.execute(
            f"select count(*) filter (where a = pa), count(*) from ("
            f"select case when affiliate_id in (9924,359,384,9452,4541,"
            f"8436,10332,3631) then 'a' || affiliate_id::varchar else "
            f"'aff_other' end a, lag(case when affiliate_id in (9924,359,"
            f"384,9452,4541,8436,10332,3631) then 'a' || affiliate_id::"
            f"varchar else 'aff_other' end) over (partition by utrip_id "
            f"order by checkin) pa from {src}) where pa is not null"
        ).fetchone()
        return {"res_count": rc, "nights": nights, "gap": gaps,
                "country": ct, "month": mon, "month_abs": mon_abs,
                "device": dev, "legs": legs,
                "repeat": (rep[0] / rep[1] if rep[1] else 0.0),
                "consec": (cons[0] / cons[1] if cons[1] else 0.0),
                "multi_trip": (tpu[0] / tpu[1] - 1 if tpu[1] else 0.0),
                "devconst": (devc[0] / devc[1] if devc[1] else 0.0),
                "affsame": (affs[0] / affs[1] if affs[1] else 0.0)}

    real_p = ROOT / "combined_set.csv"
    synth_p = ROOT / "synth" / "booking_synth.csv"
    con = duckdb.connect()
    real = mixes(con, real_p) if real_p.exists() else None
    synth = mixes(con, synth_p) if synth_p.exists() else None
    con.close()
    return CT_CELLS, Counter, ROOT, alt, ct_of, gap_cell, json, mixes, mo, \
        night_cell, pd, real, real_p, rc_bin, synth, synth_p


@app.cell
def _(mo, real, synth):
    mo.md(
        "# Synthetic vs real booking trips — scorecard\n"
        "The generator targets the **spec contract and the DP release**, not "
        "realism; gaps are declared, measured and reported, never hidden."
        + ("" if synth else "\n\n*(no synth yet — run `make synth`)*")
        + ("" if real else "\n\n*(combined_set.csv not found)*"))
    return


@app.cell
def _(alt, mo, pd, real, synth):
    _rows = []
    if real and synth:
        for _src, _d in (("real", real), ("synthetic", synth)):
            _tot = max(1, sum(_d["res_count"].values()))
            for _b, _n in _d["res_count"].items():
                _rows.append({"src": _src, "bin": _b, "share": _n / _tot})
    _df = pd.DataFrame(_rows)
    _out = mo.md("*(run `make synth`)*")
    if len(_df):
        _order = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12",
                  "rc_13p"]
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("bin:N", sort=_order), y="share:Q",
            color="src:N", xOffset="src:N").properties(
            width=480, height=220,
            title="Reservations per user: real vs synth (the DP main histogram)")
        _out = mo.ui.altair_chart(_chart)
    _out
    return


@app.cell
def _(CT_CELLS, mo, pd, real, synth):
    _score = []
    if real and synth:
        def tv(a, b, grid):
            ta, tb = max(1, sum(a.values())), max(1, sum(b.values()))
            return 0.5 * sum(abs(a.get(c, 0) / ta - b.get(c, 0) / tb)
                             for c in grid)

        _fill = {
            "res_count": "DP res_count bins (main histogram)",
            "country tier": "DP country mix + 16x16 transitions + return prior",
            "stay nights": "DP nights histogram",
            "inter-leg gap": "DP gap histogram (+ suppressed moments)",
            "checkin month": "DP month histogram (day-of-month uniform)",
            "device": "DP device histogram",
        }
        _grids = {
            "res_count": ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8",
                          "rc_9_12", "rc_13p"],
            "country tier": CT_CELLS,
            "stay nights": ["n_1", "n_2", "n_3", "n_4_6", "n_7p"],
            "inter-leg gap": ["g_0", "g_1", "g_2_3", "g_4p"],
            "checkin month": [f"{i:02d}" for i in range(1, 13)],
            "device": ["desktop", "mobile", "tablet"],
        }
        _keys = {"res_count": "res_count", "country tier": "country",
                 "stay nights": "nights", "inter-leg gap": "gap",
                 "checkin month": "month", "device": "device"}
        for _name, _grid in _grids.items():
            _t = tv(real[_keys[_name]], synth[_keys[_name]], _grid)
            _score.append({"column": _name, "kind": "TV",
                           "value": round(_t, 4),
                           "fill_source": _fill[_name]})
        _score.append({"column": "repeat-city trip share", "kind": "share diff",
                       "value": round(abs(real["repeat"] - synth["repeat"]), 4),
                       "fill_source": "DP return_rate bands + repeat_struct "
                                      "anatomy (2nd-order behavior)"})
        _score.append({"column": "consecutive same-city leg share",
                       "kind": "share diff",
                       "value": round(abs(real["consec"] - synth["consec"]), 4),
                       "fill_source": "DP repeat_struct ext-chain cells "
                                      "(A,A stay extensions; real 9.01%)"})
        _score.append({"column": "multi-trip user share",
                       "kind": "share diff",
                       "value": round(abs(real["multi_trip"] - synth["multi_trip"]), 4),
                       "fill_source": "DP trips_per_user cells (real 8.9%; "
                                      "was 0 under the single_trip prior)"})
        _score.append({"column": "single-device trip share",
                       "kind": "share diff",
                       "value": round(abs(real["devconst"] - synth["devconst"]), 4),
                       "fill_source": "DP device_persist cells (per-trip "
                                      "device plans; real 70.7%)"})
        _score.append({"column": "affiliate same-prev-tier share",
                       "kind": "share diff",
                       "value": round(abs(real["affsame"] - synth["affsame"]), 4),
                       "fill_source": "DP affiliate_persist cells (per-trip "
                                      "affiliate plans; real 75.4% at tier "
                                      "level)"})
    _df = pd.DataFrame(_score)
    _out = mo.md("*(run `make synth`)*")
    if len(_df):
        _chart = alt.Chart(_df[_df["kind"] == "TV"]).mark_bar().encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1])),
            y="column:N", color=alt.Color("fill_source:N"),
            tooltip=["column", "value", "fill_source"]).properties(
            width=520, height=180, title="Per-column TV, real vs synth")
        _out = mo.vstack([mo.md(
            "## Scorecard\n"
            "**TV = total variation distance** between the real and synthetic "
            "distributions over a column's public bins: "
            "TV = ½ · Σ|p_real(bin) − p_synth(bin)|, ranging **0 to 1** — "
            "0 means identical mixes, 1 means completely disjoint. Read it as "
            "*the share of probability mass that would have to move to make "
            "the two distributions match* (TV 0.05 ⇒ 5% of mass misplaced). "
            "The same statistic gates `make synth` validation (soft tolerance "
            "0.25) — the generator targets the DP release, so some drift from "
            "the *real* data is expected and honest.\n\n"
            "Rows marked **share diff** are plain absolute share differences "
            "of a scalar metric (not a TV over bins): the repeat-city trip "
            "share (the return_rate release, 38.1% real) and the consecutive "
            "same-city leg share (the repeat_struct ext-chain release, 9.01% "
            "real — the structure the pre-1.2.0 generator missed 5.4x)."),
            mo.ui.altair_chart(_chart),
            mo.ui.table(_df, selection=None)])
    _out
    return


@app.cell
def _(mo, pd, real, synth):
    _rows = []
    if real and synth:
        for _src, _d in (("real", real), ("synthetic", synth)):
            _tot = max(1, sum(_d["country"].values()))
            for _c, _n in _d["country"].most_common(12):
                _rows.append({"src": _src, "tier": _c, "share": _n / _tot})
    _df = pd.DataFrame(_rows)
    _out = mo.md("")
    if len(_df):
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("tier:N", sort="-y"), y="share:Q",
            color="src:N", xOffset="src:N").properties(
            width=520, height=220,
            title="Hotel-country tier mix: real vs synth")
        _out = mo.ui.altair_chart(_chart)
    _out
    return


@app.cell
def _(CT_CELLS, alt, mo, pd, real, synth):
    _out = mo.md("")
    if real and synth:
        # per-tier share delta — shows the country-tier TV is DIFFUSE, not a
        # broken tier (pre-1.2.0 max |delta| was 0.008)
        _tr = sum(real["country"].values())
        _ts = sum(synth["country"].values())
        _rows = []
        for _c in CT_CELLS:
            _dr = real["country"].get(_c, 0) / _tr
            _ds = synth["country"].get(_c, 0) / _ts
            _rows.append({"tier": _c, "real": round(_dr, 4),
                          "synth": round(_ds, 4),
                          "delta": round(_ds - _dr, 4)})
        _df = pd.DataFrame(_rows).sort_values("delta", key=abs,
                                              ascending=False)
        _out = mo.vstack([mo.md(
            "## Per-tier share delta (synth − real)\n"
            "The country-tier TV is spread over 16 cells; this table shows "
            "no single tier is broken — large |delta| here would point at "
            "the return-boost diagonal over-firing on a specific tier."),
            mo.ui.table(_df, selection=None)])
    _out
    return


@app.cell
def _(alt, mo, pd, real, synth):
    _out = mo.md("")
    if real and synth:
        # month-of-WINDOW placement: the real observation ramp vs the drawn
        # starts (month-of-year TV hides the censoring shape entirely)
        _mr = sum(real["month_abs"].values())
        _ms = sum(synth["month_abs"].values())
        _months = sorted(set(real["month_abs"]) | set(synth["month_abs"]))
        _rows = []
        for _src, _d, _t in (("real", real, _mr), ("synthetic", synth, _ms)):
            for _m in _months:
                _rows.append({"src": _src, "month": _m,
                              "share": _d["month_abs"].get(_m, 0) / _t})
        _df = pd.DataFrame(_rows)
        _chart = alt.Chart(_df).mark_line(point=True).encode(
            x=alt.X("month:N", sort=_months), y="share:Q",
            color="src:N").properties(
            width=560, height=240,
            title="Start-month-of-window: the real observation ramp vs synth "
                  "(the 1.2.0 month_window release)")
        _out = mo.vstack([mo.md(
            "## Calendar placement (month-of-window)\n"
            "The real data is an observation RAMP (peak 2016-08 19.5% → "
            "2017-02 1.5%) shaped by window censoring — not flat seasonality. "
            "Spec 1.2.0 releases month-of-window so synthetic starts follow "
            "the true shape (pre-1.2.0 month-of-year draws put 0.01% of "
            "starts in 2017-02 vs the real 1.5%)."),
            mo.ui.altair_chart(_chart)])
    _out
    return


@app.cell
def _(alt, mo, pd, real, synth):
    _out = mo.md("")
    if real and synth:
        # exact legs-per-trip: exposes WITHIN-bin drift that the public rc
        # bins cannot see (the within_bin_decay prior targets this)
        _tr = sum(real["legs"].values())
        _ts = sum(synth["legs"].values())
        _rows = []
        for _src, _d, _t in (("real", real, _tr), ("synthetic", synth, _ts)):
            for _l in range(1, 15):
                _rows.append({"src": _src, "legs": _l,
                              "share": _d["legs"].get(_l, 0) / _t})
        _df = pd.DataFrame(_rows)
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("legs:O"), y="share:Q", color="src:N",
            xOffset="src:N").properties(
            width=560, height=240,
            title="Legs per trip, exact counts (the within-bin view)")
        _out = mo.vstack([mo.md(
            "## Legs per trip — exact counts\n"
            "The public rc bins (4 / 5 / 6 / 7-8 / 9-12 / 13+) bin this "
            "away; the exact view is where uniform-within-bin draws used to "
            "over-fill the 10-12 range. Spec 1.2.0 draws inside multi-width "
            "bins with the declared geometric decay (real within-bin mass "
            "decays ~0.7 per step)."),
            mo.ui.altair_chart(_chart)])
    _out
    return


@app.cell
def _(mo, synth_p):
    _out = mo.md("*(run `make synth`)*")
    if synth_p.exists():
        import duckdb as _duckdb_mod
        _con = _duckdb_mod.connect()
        _top = _con.execute(
            f"select prev_city || '->' || next_city corridor, n_trans "
            f"from (select city_a prev_city, city_b next_city, count(*) "
            f"n_trans from (select utrip_id, city_id city_a, "
            f"lead(city_id) over (partition by utrip_id order by checkin) "
            f"city_b from read_csv_auto('{synth_p}')) "
            f"where city_b is not null group by 1, 2) "
            f"order by n_trans desc limit 10").df()
        _con.close()
        _out = mo.vstack([
            mo.md("## Top synthetic city corridors (sequence structure)\n"
                  "Synth city pairs come ONLY from the released top-k list "
                  "(suppression ≥ 50 users / ≥ 100 transitions); rare real "
                  "corridors cannot leak because they were never released."),
            mo.ui.table(_top, selection=None)])
    _out
    return


@app.cell
def _(ROOT, json, mo):
    _items = []
    for _name, _path in [
        ("validation", ROOT / "synth" / "validation.json"),
        ("replicates", ROOT / "synth" / "replicates.json"),
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
