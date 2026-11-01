# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb"]
# ///
"""Enclave EDA notebook: travel volumes, the heavy-tailed travels-per-user
distribution, inter-travel gaps and their moments, motifs, sequence structure,
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
        # Enclave travel-sequence encoder — EDA
        84,964 international transfers, 29,640 users, 77 countries
        (2012-04-04 → 2013-09-16, Foursquare TIST2015 derived). All plots in
        this notebook are **enclave-only**; the final sections show the ε-DP
        release and the formal layer.
        """
    )
    return


@app.cell
def _(alt, con, mo):
    _tc = con.execute(
        "select tc_bin, count(*) n from feat_users group by 1").df()
    _order = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16", "tc_17_32", "tc_33p"]
    _chart = alt.Chart(_tc).mark_bar().encode(
        x=alt.X("tc_bin:N", sort=_order, title="travels per user (public bins)"),
        y=alt.Y("n:Q", title="users")).properties(
        width=480, height=240,
        title="The heavy tail: users per travel-count bin (median 2, max 195)")
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(alt, con, mo):
    _g = con.execute(
        "select gap_s/86400.0 gap_days from feat_transfers "
        "where gap_s is not null").df()
    _hist = alt.Chart(_g).mark_bar().encode(
        x=alt.X("gap_days:Q", bin=alt.Bin(maxbins=60),
                scale=alt.Scale(type="symlog"), title="inter-travel gap (days)"),
        y="count()").properties(width=640, height=220,
                                title="Inter-travel gap distribution")
    _m = con.execute(
        "select gap_class, n_gaps, round(mean_days,1) mean_days, "
        "round(median_days,1) median_days, round(sd_days,1) sd_days "
        "from mart_gap_moments order by mean_days").df()
    mo.vstack([
        mo.ui.altair_chart(_hist),
        mo.md("**Gap moments per class** (heavy right skew — mean 40.7 d vs "
              "median 12.9 d overall; the suppressed-aggregate release)"),
        mo.ui.table(_m, selection=None),
    ])
    return


@app.cell
def _(alt, con, mo):
    _t = con.execute(
        "select prev_region, next_region, n, round(lift,2) lift "
        "from mart_seq_transitions").df()
    _heat = alt.Chart(_t).mark_rect().encode(
        x=alt.X("next_region:N", title="to region"),
        y=alt.Y("prev_region:N", title="from region"),
        color=alt.Color("n:Q", scale=alt.Scale(type="log", scheme="blues")),
        tooltip=["prev_region", "next_region", "n", "lift"]).properties(
        width=320, height=320, title="Region transition matrix (exact, enclave)")
    mo.ui.altair_chart(_heat)
    return


@app.cell
def _(con, mo):
    _corr = con.execute(
        "select prev_country a, next_country b, n_repeats, "
        "round(cv,2) cv from mart_motif_corridors where is_corridor "
        "order by n_repeats desc limit 10").df()
    _tours = con.execute(
        "select n_countries, count(*) n from mart_motif_tours where is_tour "
        "group by 1 order by 1").df()
    _ret = con.execute(
        "select count(*) filter (is_return) r, count(*) n "
        "from mart_motif_returns").fetchone()
    mo.vstack([
        mo.md(f"## Motifs (public thresholds; SQL verdicts re-decided in Lean)\n"
              f"Return trips: **{_ret[0]:,} / {_ret[1]:,}** consecutive pairs "
              f"({100*_ret[0]/max(1,_ret[1]):.1f}%) — the dominant structure."),
        mo.hstack([
            mo.vstack([mo.md("**Top commuter corridors (≥5 reps, CV ≤ 0.8)** "
                             "— user ids withheld"),
                       mo.ui.table(_corr, selection=None)]),
            mo.vstack([mo.md("**Tours (≥3 hops ≤7d, ≥3 countries)**"),
                       mo.ui.table(_tours, selection=None)]),
        ]),
    ])
    return


@app.cell
def _(alt, con, mo):
    _e = con.execute(
        "select ep_bin, count(*) n from feat_episodes group by 1").df()
    _d = con.execute(
        "select dur_class, count(*) n from feat_transfers group by 1").df()
    _ep = alt.Chart(_e).mark_bar().encode(
        x=alt.X("ep_bin:N", sort=["ep_1", "ep_2", "ep_3", "ep_4_5", "ep_6p"]),
        y="n:Q").properties(width=300, height=200,
                            title="Journey-episode lengths")
    _du = alt.Chart(_d).mark_bar().encode(
        x=alt.X("dur_class:N", sort=["dur_day", "dur_week", "dur_month",
                                     "dur_quarter", "dur_long"]),
        y="n:Q").properties(width=300, height=200, title="Stay durations")
    mo.ui.altair_chart(_ep | _du)
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
            "select * from mart_dp_travel_count").fetchall()}
        _rows = [{"bin": c["cell"], "value": int(c["noised_count"]),
                  "kind": "released (ε-DP)"} for c in _rel["cells"]]
        _rows += [{"bin": c, "value": n, "kind": "clipped (pre-noise, enclave)"}
                  for c, n in _clipped.items()]
        _order = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16",
                  "tc_17_32", "tc_33p"]
        _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
            x=alt.X("bin:N", sort=_order),
            y="value:Q", color="kind:N", xOffset="kind:N").properties(
            width=480, height=240,
            title="DP release utility: travels-per-user, released vs pre-noise")
        _led = _rel["budget"]
        _out = mo.vstack([
            mo.md(f"## The ε-DP release (what actually leaves)\n"
                  f"Privacy unit: **one user** (n = 29,640 — SNR is comfortable; "
                  f"the heavy tail is clipped at L1 = 8). Ledger: "
                  f"**{_led['spent_num']}/{_led['spent_den']}** of 4/1, "
                  f"exactly saturated. The travel-count histogram has "
                  f"sensitivity 1 (each user in exactly one bin) — compare bar "
                  f"pairs to judge SNR honestly."),
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
                  for k in ("returns", "corridors", "tours", "bursts",
                            "longstays", "assoc")}
        _out = mo.md(
            f"## Formal layer\n"
            f"Claims: **{_proved} proved**, {_assumed} assumed (trust boundary "
            f"stated explicitly). SQL↔Lean agreement: `{_agree}`. Episodes all "
            f"gap-chained: "
            f"{all(s['lean_is_episode'] for s in _res['episodes'])}.")
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
        verified DP noise). The library `lean/EnclaveTravel/` holds the
        definitions and proofs; the repo-root `lean/*.lean` files are the
        executables `lake` links against it:

        | File | What it formalizes |
        |---|---|
        | `EnclaveTravel/Params.lean` | the concept spec mirrored from dbt `vars:` + the ISO→region map (locked by `tests/test_spec_sync.py`) |
        | `EnclaveTravel/Motifs.lean` | **decidable** motif predicates — `IsReturnPair`, `IsCorridor` (repeat count ≥ 5 AND gap CV ≤ 4/5, as the exact `Nat` inequality 25·k·s₂ ≤ 41·s₁², no floats), `IsTour`, `IsBurst`, `IsLongStay` — with canonical-example and monotonicity theorems |
        | `EnclaveTravel/Sequences.lean` | `IsEpisode`: gap-chained, chronologically ordered journeys (tail/mono/ordered lemmas) |
        | `EnclaveTravel/Stats.lean` | clipped per-user cell sums; `clippedSum_sensitivity` (add/remove one user moves a cell by ≤ clip) |
        | `EnclaveTravel/DPRelease.lean` | the release mechanism on SampCert discrete Laplace: `privClippedCell_PureDP` (per cell) and `histMech_DP` (the composed K-cell histogram is K·ε₁/ε₂-DP) |
        | `EnclaveTravel/Sampler.lean` | the verified categorical sampler; `sampleCellPMF_law`: P(i) = wᵢ/Σw, exactly |
        | `EnclaveTravel/Assoc.lean` | χ² recomputed over ℚ; `chiSq_nonneg`, `chiSq_eq_zero_of_indep` |
        | `EnclaveTravel/GenSpec.lean` | the generator's typed spec + smart constructors with validity **by construction**: timestamps strictly increasing (`List.Pairwise`), prev ≠ next country, duration chaining, window containment |
        | `EnclaveTravel/TravelShape.lean` | the public calendar (kernel-checked date anchors), country centroids, 24-hex venue rendering |
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
        _files = (sorted((ROOT / "lean" / "EnclaveTravel").glob("*.lean"))
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
