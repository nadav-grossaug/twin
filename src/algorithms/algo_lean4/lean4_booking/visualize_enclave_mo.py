# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb"]
# ///
"""Enclave EDA notebook: trip structure (legs, cities, countries), the
contiguous temporal flow, stay nights and gaps, motifs (repeat-city returns,
corridors, tours, long stays), the chi2 feature associations, seasonality,
and DP-release utility. Everything rendered here is ENCLAVE-SIDE (per-user
detail allowed); only the DP release section shows what actually leaves."""

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
        # Enclave trip encoder — EDA
        1,545,502 reservations, 259,159 users, 288,348 multi-destination
        trips, 39,901 cities, 195 hotel countries (2015-12-31 → 2017-02-27,
        Booking.com WSDM'21 multi-destination trips). All plots in this
        notebook are **enclave-only**; the final sections show the ε-DP
        release and the formal layer.
        """
    )
    return


@app.cell
def _(alt, con, mo):
    _legs = con.execute(
        "select n_legs, count(*) n from feat_trips group by 1 order by 1").df()
    _chart = alt.Chart(_legs).mark_bar().encode(
        x=alt.X("n_legs:Q", title="legs per trip (median 5, max 48)"),
        y=alt.Y("n:Q", title="trips", scale=alt.Scale(type="log"))).properties(
        width=480, height=240,
        title="Trip length: nearly all trips have >= 4 legs")
    _ret = con.execute(
        "select count(*) filter (where has_repeat_city) r, count(*) n "
        "from feat_trips").fetchone()
    _cty = con.execute(
        "select n_countries, count(*) n from feat_trips group by 1 "
        "order by 1").df()
    _c2 = alt.Chart(_cty).mark_bar().encode(
        x=alt.X("n_countries:Q", title="distinct hotel countries per trip"),
        y="n:Q").properties(width=480, height=200,
                            title="Country spread per trip (72% stay in 1 country)")
    mo.vstack([
        mo.ui.altair_chart(_chart),
        mo.md(f"**Repeat-city returns: {_ret[0]:,} / {_ret[1]:,} trips "
              f"({100*_ret[0]/max(1,_ret[1]):.1f}%) revisit a city within the "
              f"same trip** — the dominant motif."),
        mo.ui.altair_chart(_c2),
    ])
    return


@app.cell
def _(alt, con, mo):
    _g = con.execute(
        "select gap_days from feat_transitions where gap_days >= 0").df()
    _hist = alt.Chart(_g).mark_bar().encode(
        x=alt.X("gap_days:Q", bin=alt.Bin(maxbins=40),
                title="days between checkout and the next checkin"),
        y="count()").properties(width=640, height=220,
                                title="Inter-leg gaps (99.1% are 0 — contiguous trips)")
    _m = con.execute(
        "select gap_class, n_gaps, round(mean_days,1) mean_days, "
        "round(median_days,1) median_days, round(sd_days,1) sd_days "
        "from mart_gap_moments order by mean_days").df()
    _n = con.execute(
        "select nights, count(*) n from stg_reservations group by 1 "
        "order by 1").df()
    _nh = alt.Chart(_n).mark_bar().encode(
        x=alt.X("nights:Q", title="stay nights (median 1, max 30)"),
        y=alt.Y("n:Q", scale=alt.Scale(type="log"))).properties(
        width=320, height=220, title="Stay nights")
    mo.vstack([
        mo.ui.altair_chart(_hist),
        mo.hstack([mo.ui.altair_chart(_nh),
                   mo.vstack([mo.md("**Gap moments per class** "
                                    "(the suppressed-aggregate release)"),
                              mo.ui.table(_m, selection=None)])]),
    ])
    return


@app.cell
def _(alt, con, mo):
    _t = con.execute(
        "select prev_ct, next_ct, n, round(lift,2) lift "
        "from mart_seq_transitions").df()
    _heat = alt.Chart(_t).mark_rect().encode(
        x=alt.X("next_ct:N", title="to country tier"),
        y=alt.Y("prev_ct:N", title="from tier"),
        color=alt.Color("n:Q", scale=alt.Scale(type="log", scheme="blues")),
        tooltip=["prev_ct", "next_ct", "n", "lift"]).properties(
        width=380, height=380,
        title="Country-tier transition matrix (exact, enclave) — the "
              "next-destination signal the DP release discretizes")
    mo.ui.altair_chart(_heat)
    return


@app.cell
def _(con, mo):
    _corr = con.execute(
        "select prev_country, next_country, n_repeats, round(cv,2) cv "
        "from mart_motif_corridors where is_corridor "
        "order by n_repeats desc limit 10").df()
    _tours = con.execute(
        "select n_countries, count(*) n from mart_motif_tours where is_tour "
        "group by 1 order by 1").df()
    _ls = con.execute(
        "select count(*) n from mart_motif_longstays where is_longstay"
    ).fetchone()[0]
    mo.vstack([
        mo.md("## Motifs (public thresholds; SQL verdicts re-decided in Lean)"),
        mo.hstack([
            mo.vstack([mo.md("**Corridors (repeated ordered country pairs)** "
                             "— user ids withheld"),
                       mo.ui.table(_corr, selection=None)]),
            mo.vstack([mo.md(f"**Tours (>=3 countries)** and "
                             f"**long stays (>=7 nights): {_ls:,} legs**"),
                       mo.ui.table(_tours, selection=None)]),
        ]),
    ])
    return


@app.cell
def _(alt, con, mo):
    _dev = con.execute(
        "select device_class, count(*) n from stg_reservations "
        "group by 1 order by 2 desc").df()
    _bk = con.execute(
        "select booker_country, count(*) n from feat_users group by 1 "
        "order by 2 desc").df()
    _mon = con.execute(
        "select month_cell, count(*) n from stg_reservations group by 1 "
        "order by 1").df()
    _d = alt.Chart(_dev).mark_bar().encode(x="device_class:N", y="n:Q").properties(
        width=240, height=200, title="Device mix")
    _b = alt.Chart(_bk).mark_bar().encode(
        x=alt.X("booker_country:N", sort="-y"), y="n:Q").properties(
        width=280, height=200, title="Booker countries (5, anonymized)")
    _m = alt.Chart(_mon).mark_bar().encode(x="month_cell:N", y="n:Q").properties(
        width=380, height=200, title="Checkin seasonality (DP month cells)")
    mo.vstack([mo.md("## Margins"), mo.ui.altair_chart(_d | _b),
               mo.ui.altair_chart(_m)])
    return


@app.cell
def _(con, mo):
    _assoc = con.execute(
        "select feat_a, feat_b, round(chi2,1) chi2, dof, "
        "round(cramers_v,3) cramers_v from mart_feature_assoc "
        "order by cramers_v desc").df()
    mo.vstack([
        mo.md("## Feature associations (χ² recomputed exactly over ℚ in Lean)"),
        mo.ui.table(_assoc, selection=None),
    ])
    return


@app.cell
def _(ROOT, alt, con, json, mo, pd):
    _rel_p = ROOT / "lean" / "out" / "dp_release.json"
    _out = mo.md("*(no DP release yet — run `make train`)*")
    if _rel_p.exists():
        _rel = json.loads(_rel_p.read_text())
        _clipped = {c: int(n) for c, n, _, _ in con.execute(
            "select * from mart_dp_res_count").fetchall()}
        _rows = [{"bin": c["cell"], "value": int(c["noised_count"]),
                  "kind": "released (ε-DP)"} for c in _rel["cells"]]
        _rows += [{"bin": c, "value": n, "kind": "clipped (pre-noise, enclave)"}
                  for c, n in _clipped.items()]
        _order = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12",
                  "rc_13p"]
        _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
            x=alt.X("bin:N", sort=_order),
            y="value:Q", color="kind:N", xOffset="kind:N").properties(
            width=480, height=240,
            title="DP release utility: reservations-per-user, released vs pre-noise")
        _led = _rel["budget"]
        _aux = {h["name"]: len(h["cells"]) for h in _rel["aux"]}
        _out = mo.vstack([
            mo.md(f"## The ε-DP release (what actually leaves)\n"
                  f"Privacy unit: **one user** (n = 259,159 — SNR is "
                  f"comfortable at ε≈10.08, a deliberately WEAK bound the user "
                  f"chose for fidelity over privacy; do not call the release "
                  f"private). Ledger: **{_led['spent_num']}/"
                  f"{_led['spent_den']}** of {_led.get('total_num', 14)}/"
                  f"{_led.get('total_den', 1)}. The res-count histogram has "
                  f"sensitivity 1 (each user in exactly one bin); aux grids: "
                  f"`{_aux}` — the 16×16 country-transition grid is the "
                  f"distilled NextCountry teacher."),
            mo.ui.altair_chart(_chart),
        ])
    _out
    return


@app.cell
def _(ROOT, json, mo):
    _p = ROOT / "lean" / "out" / "lean_results.json"
    _out = mo.md("*(run `make train`)*")
    if _p.exists():
        _res = json.loads(_p.read_text())
        _proved = sum(1 for c in _res["claims"] if c["status"] == "proved")
        _assumed = sum(1 for c in _res["claims"] if c["status"] == "assumed")
        _agree = {k: f"{sum(1 for x in _res[k] if x.get('agree'))}/{len(_res[k])}"
                  for k in ("returns", "corridors", "tours", "longstays",
                            "assoc", "trips")}
        _out = mo.md(
            f"## Formal layer\n"
            f"Claims: **{_proved} proved**, {_assumed} assumed (trust boundary "
            f"stated explicitly). SQL↔Lean agreement: `{_agree}`.")
    _out
    return


@app.cell
def _(ROOT, json, mo, pd):
    _p = ROOT / "lean" / "out" / "lean_results.json"
    _claims = (json.loads(_p.read_text())["claims"] if _p.exists() else [])
    _intro = mo.md(
        """
        ## The Lean 4 formulation

        The formal layer lives in `lean/` and is built with the pinned
        toolchain (Lean v4.34.1, Mathlib + **SampCert @ a2c53bb** for the
        verified DP noise). The library `lean/EnclaveTrip/` holds the
        definitions and proofs; the repo-root `lean/*.lean` files are the
        executables `lake` links against it:

        | File | What it formalizes |
        |---|---|
        | `EnclaveTrip/Params.lean` | the concept spec mirrored from dbt `vars:` + the country-tier / affiliate-tier grids (locked by `tests/test_spec_sync.py`) |
        | `EnclaveTrip/Motifs.lean` | **decidable** motif predicates — `HasRepeatCity`, `IsCorridor` (repeat count AND gap-CV as the exact `Nat` inequality, no floats), `IsTour`, `IsLongStay` — with canonical-example theorems |
        | `EnclaveTrip/Sequences.lean` | `IsWellFormedTrip` / `IsContiguousTrip`: strictly increasing checkins, no overlapping presence, checkout→next-checkin chaining (the 99.1% temporal-flow property) |
        | `EnclaveTrip/Stats.lean` | clipped per-user cell sums; `clippedSum_sensitivity` (add/remove one user moves a cell by ≤ clip) |
        | `EnclaveTrip/DPRelease.lean` | the release mechanism on SampCert discrete Laplace: `privClippedCell_PureDP` (per cell) and `histMech_DP` (the composed K-cell histogram is K·ε₁/ε₂-DP) |
        | `EnclaveTrip/Sampler.lean` | the verified categorical sampler; `sampleCellPMF_law`: P(i) = wᵢ/Σw, exactly |
        | `EnclaveTrip/Assoc.lean` | χ² recomputed over ℚ; `chiSq_nonneg`, `chiSq_eq_zero_of_indep` |
        | `EnclaveTrip/GenSpec.lean` | the generator's typed spec + smart constructors with validity **by construction**: increasing checkin days (`List.Pairwise`), nights ≥ 1, window containment |
        | `EnclaveTrip/TripShape.lean` | the public calendar (kernel-checked date anchors), city id rendering |
        | `Release.lean` / `Checks.lean` / `Generate.lean` / `Sample.lean` / `Catsample.lean` | executables: ε-DP release (exact-fraction ledger, exit 2 over budget), SQL↔Lean re-deciding, the standalone synthetic generator, sampler tools |

        A failing proof fails `lake build`, so nothing runs unverified — the
        same sources ship in the bundle, where the consumer's kernel
        re-checks every theorem before the generator is even linked.
        """)
    _ledger = mo.md("*(claims ledger appears after `make train`)*")
    if _claims:
        _ledger = mo.vstack([
            mo.md("**Proofs and lemmas** (the claims ledger `lake exe checks` "
                  "emits; `assumed` rows state trust boundary 2 — Lean proves "
                  "the mechanism and predicates, not the SQL that fed them):"),
            mo.ui.table(pd.DataFrame(_claims)[["name", "status", "statement"]],
                        selection=None, page_size=30),
        ])
    mo.vstack([_intro, _ledger])
    return


@app.cell
def _(mo):
    show_lean_src = mo.ui.switch(label="show Lean 4 source code")
    show_lean_src
    return (show_lean_src,)


@app.cell
def _(ROOT, mo, show_lean_src):
    _out = mo.md("")
    if show_lean_src.value:
        _files = (sorted((ROOT / "lean" / "EnclaveTrip").glob("*.lean"))
                  + [ROOT / "lean" / n for n in
                     ("Release.lean", "Checks.lean", "Generate.lean",
                      "Sample.lean", "Catsample.lean")])
        _out = mo.accordion({
            str(_f.relative_to(ROOT / "lean")):
                mo.md(f"```lean\n{_f.read_text()}\n```")
            for _f in _files if _f.exists()
        })
    _out
    return


if __name__ == "__main__":
    app.run()
