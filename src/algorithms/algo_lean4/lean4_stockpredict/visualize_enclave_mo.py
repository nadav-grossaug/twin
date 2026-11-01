# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb"]
# ///
"""Enclave EDA notebook: prices, log returns, moments, seasonality,
collinearity, GARCH footprint, motifs, and DP-release utility.
Everything rendered here is ENCLAVE-SIDE (per-ticker detail allowed);
only the DP release section shows what actually leaves."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import pandas as pd

    alt.data_transformers.enable("default", max_rows=20_000)
    ROOT = Path(__file__).resolve().parent
    con = duckdb.connect(str(ROOT / "artifacts" / "enclave.duckdb"),
                         read_only=True)
    return ROOT, alt, con, json, mo, pd


@app.cell
def _(mo):
    mo.md(
        """
        # Enclave stock-series encoder — EDA
        8 tickers x 168 trading days, base-100 normalized. Log returns are the
        modelling basis (random walk in log space). All plots in this notebook
        are **enclave-only**; the final section shows the ε-DP release.
        """
    )
    return


@app.cell
def _(alt, con, mo):
    _px = con.execute(
        "select ticker, d, price from feat_returns order by ticker, d").df()
    _chart = alt.Chart(_px).mark_line().encode(
        x="d:T", y=alt.Y("price:Q", title="price (base 100)"),
        color="ticker:N").properties(width=640, height=280,
                                     title="Normalized price paths")
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(con, mo):
    _stats = con.execute(
        "select ticker, n_days, round(mean_bp,1) mean_bp, round(sd_bp,1) sd_bp, "
        "round(skew,2) skew, round(kurtosis_excess,2) kurt, "
        "round(autocorr1,3) ac1, round(abs_autocorr1,3) abs_ac1, "
        "min_bp, max_bp from mart_series_stats order by ticker").df()
    mo.vstack([
        mo.md("## Statistical moments of daily log returns (bp)\n"
              "`ac1` ≈ 0 is the random-walk check; `abs_ac1` > 0 is the "
              "volatility-clustering (GARCH) footprint."),
        mo.ui.table(_stats, selection=None),
    ])
    return


@app.cell
def _(alt, con, mo):
    _r = con.execute(
        "select ret_bp, ticker from feat_returns where ret_bp is not null").df()
    _hist = alt.Chart(_r).mark_bar().encode(
        x=alt.X("ret_bp:Q", bin=alt.Bin(maxbins=60), title="daily log return (bp)"),
        y="count()", color="ticker:N").properties(width=640, height=220,
                                                  title="Return distribution")
    mo.ui.altair_chart(_hist)
    return


@app.cell
def _(alt, con, mo):
    _s = con.execute("select * from mart_seasonality").df()
    _dow = alt.Chart(_s[_s.dimension == "dow"]).mark_bar().encode(
        x=alt.X("bucket:N", sort=["mon", "tue", "wed", "thu", "fri"]),
        y=alt.Y("mean_bp:Q", title="mean return (bp)")).properties(
        width=300, height=200, title="Seasonality: day of week")
    _mon = alt.Chart(_s[_s.dimension == "month"]).mark_bar().encode(
        x="bucket:N", y="mean_bp:Q").properties(
        width=300, height=200, title="Seasonality: month")
    mo.ui.altair_chart(_dow | _mon)
    return


@app.cell
def _(alt, con, mo):
    _c = con.execute(
        "select ticker_a, ticker_b, pearson_r from mart_correlation").df()
    _sym = _c.rename(columns={"ticker_a": "a", "ticker_b": "b"})
    _sym2 = _sym.rename(columns={"a": "b", "b": "a"})
    import pandas as _pd
    _full = _pd.concat([_sym, _sym2])
    _heat = alt.Chart(_full).mark_rect().encode(
        x="a:N", y="b:N",
        color=alt.Color("pearson_r:Q", scale=alt.Scale(scheme="blueorange",
                                                       domain=[-1, 1])),
        tooltip=["a", "b", "pearson_r"]).properties(
        width=320, height=320, title="Inter-stock collinearity (Pearson r)")
    mo.ui.altair_chart(_heat)
    return


@app.cell
def _(ROOT, json, mo):
    _reg = ROOT / "models_out" / "registry.json"
    _rows = []
    if _reg.exists():
        _active = json.loads(_reg.read_text())["minimodels"]["active"]
        _meta = json.loads((ROOT / "models_out" / "minimodels" / _active /
                            "meta.json").read_text())
        for _t, _p in _meta["garch"]["per_ticker"].items():
            _rows.append({"ticker": _t, "alpha": round(_p["alpha"], 3),
                          "beta": round(_p["beta"], 3),
                          "persistence": round(_p["persistence"], 3),
                          "adopted": _p["adopted"]})
    import pandas as _pd
    mo.vstack([
        mo.md("## GARCH(1,1) per ticker (PyTorch MLE, enclave-only)\n"
              "Persistence α+β near 1 = volatility clusters. These parameters "
              "are PRIVATE; the released counterpart is the vol-state "
              "transition histogram."),
        mo.ui.table(_pd.DataFrame(_rows), selection=None),
    ])
    return


@app.cell
def _(con, mo):
    _runs = con.execute(
        "select ticker, start_d, end_d, len_days, sgn, total_bp from "
        "mart_motif_runs where is_run order by len_days desc limit 10").df()
    _shocks = con.execute(
        "select ticker, d, ret_bp from mart_motif_shock where is_shock "
        "order by abs(ret_bp) desc limit 10").df()
    _dd = con.execute(
        "select ticker, round(max_dd_frac,3) max_dd, n_days_in_drawdown, "
        "worst_day from mart_motif_drawdown order by max_dd_frac desc").df()
    mo.vstack([
        mo.md("## Motifs (public thresholds; SQL verdicts re-decided in Lean)"),
        mo.hstack([
            mo.vstack([mo.md("**Runs ≥ 5 days**"), mo.ui.table(_runs, selection=None)]),
            mo.vstack([mo.md("**Shock days ≥ 10%**"), mo.ui.table(_shocks, selection=None)]),
        ]),
        mo.vstack([mo.md("**Drawdowns ≥ 20%**"), mo.ui.table(_dd, selection=None)]),
    ])
    return


@app.cell
def _(ROOT, alt, con, json, mo, pd):
    _rel_p = ROOT / "lean" / "out" / "dp_release.json"
    _out = mo.md("*(no DP release yet — run `make lean`)*")
    if _rel_p.exists():
        _rel = json.loads(_rel_p.read_text())
        _clipped = {c: int(n) for c, n, _, _ in con.execute(
            "select * from mart_dp_counts").fetchall()}
        _rows = [{"class": c["cell"], "value": int(c["released"]),
                  "kind": "released (ε-DP)"} for c in _rel["cells"]]
        _rows += [{"class": c, "value": n, "kind": "clipped (pre-noise, enclave)"}
                  for c, n in _clipped.items()]
        _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
            x=alt.X("class:N", sort=["crash", "down", "flat", "up", "surge"]),
            y="value:Q", color="kind:N", xOffset="kind:N").properties(
            width=480, height=240,
            title="DP release utility: class mix, released vs pre-noise")
        _led = _rel["ledger"]
        _out = mo.vstack([
            mo.md(f"## The ε-DP release (what actually leaves)\n"
                  f"Privacy unit: **one ticker** (n = 8 — the dominant "
                  f"calibration risk). Ledger: "
                  f"**{_led['total_eps_num']}/{_led['total_eps_den']}** of 4/1, "
                  f"exactly saturated. Noise scale on the class mix = "
                  f"clip·ε₂/ε₁ = 128·50/27 ≈ 237 — compare bar pairs to judge "
                  f"SNR honestly."),
            mo.ui.altair_chart(_chart),
        ])
    _out
    return


@app.cell
def _(ROOT, json, mo):
    _p = ROOT / "lean" / "out" / "lean_results.json"
    _out = mo.md("*(run `make lean-checks`)*")
    if _p.exists():
        _res = json.loads(_p.read_text())
        _proved = sum(1 for c in _res["claims"] if c["status"] == "proved")
        _assumed = sum(1 for c in _res["claims"] if c["status"] == "assumed")
        _agree = {k: f"{sum(1 for x in _res[k] if x.get('agree'))}/{len(_res[k])}"
                  for k in ("runs", "shocks", "drawdowns", "vrecoveries",
                            "volbursts", "assoc")}
        _out = mo.md(
            f"## Formal layer\n"
            f"Claims: **{_proved} proved**, {_assumed} assumed (trust boundary "
            f"stated explicitly). SQL↔Lean agreement: `{_agree}`. Sessions all "
            f"timely: "
            f"{all(s['lean_is_timely_session'] for s in _res['sessions'])}.")
    _out
    return


if __name__ == "__main__":
    app.run()
