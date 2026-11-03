# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb"]
# ///
"""Enclave EDA notebook: the social graph's degree distribution, local
clustering (the published 0.6055 average), triangle heavy tail, community
structure and mixing, wedge closure, feature categories, circles, motifs,
chi-square associations, and DP-release utility. Everything rendered here is
ENCLAVE-SIDE (per-node detail allowed); only the DP release section shows
what actually leaves."""

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
        # Enclave social-graph encoder — EDA
        4,039 people, 88,234 mutual friendships (SNAP "Social circles:
        Facebook", 10 ego networks, 2,280 ego-scoped anonymized features,
        friendship circles). The graph is a **static snapshot — no temporal
        fields** (verified by `q_temporal_fields`). All plots in this
        notebook are **enclave-only**; the final sections show the ε-DP
        release and the formal layer.
        """
    )
    return


@app.cell
def _(alt, con, mo):
    _deg = con.execute(
        "select degree, count(*) n from feat_nodes group by 1").df()
    _loglog = alt.Chart(_deg).mark_point(size=18, opacity=0.6).encode(
        x=alt.X("degree:Q", scale=alt.Scale(type="log"), title="degree"),
        y=alt.Y("n:Q", scale=alt.Scale(type="log"), title="nodes")).properties(
        width=330, height=240, title="Degree distribution (log-log)")
    _bins = con.execute(
        "select degree_class, n_nodes from mart_degree_stats").df()
    _order = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16",
              "deg_17_32", "deg_33_64", "deg_65p"]
    _bar = alt.Chart(_bins).mark_bar().encode(
        x=alt.X("degree_class:N", sort=_order, title="public degree bins"),
        y=alt.Y("n_nodes:Q", title="nodes")).properties(
        width=330, height=240,
        title="Nodes per public degree bin (median 25, max 1045)")
    mo.ui.altair_chart(_loglog | _bar)
    return


@app.cell
def _(alt, con, mo):
    _cc = con.execute(
        "select cc_num * 1.0 / nullif(cc_den, 0) cc from feat_nodes").df()
    _stats = con.execute("select * from mart_triangle_stats").df().iloc[0]
    _hist = alt.Chart(_cc.fillna(0.0)).mark_bar().encode(
        x=alt.X("cc:Q", bin=alt.Bin(maxbins=40),
                title="local clustering coefficient"),
        y="count()").properties(width=340, height=220,
                                title="Local clustering distribution")
    _classes = con.execute(
        "select cc_class, n_nodes from mart_clustering").df()
    _corder = ["cc_zero", "cc_0_20", "cc_20_40", "cc_40_60", "cc_60_80",
               "cc_80_99", "cc_one"]
    _cbar = alt.Chart(_classes).mark_bar().encode(
        x=alt.X("cc_class:N", sort=_corder, title="public clustering classes"),
        y="n_nodes:Q").properties(width=340, height=220,
                                  title="Nodes per clustering class")
    mo.vstack([
        mo.md(f"## Triadic closure — the dataset's signature\n"
              f"Average local clustering **{_stats['avg_clustering']}** "
              f"(published value ≈ 0.6055 — reproduced), transitivity "
              f"**{_stats['transitivity']}**: "
              f"{int(_stats['total_triangles']):,} triangles over "
              f"{int(_stats['total_wedges']):,} wedges."),
        mo.ui.altair_chart(_hist | _cbar),
    ])
    return


@app.cell
def _(alt, con, mo):
    _tri = con.execute(
        "select tri_class, count(*) n from feat_nodes group by 1").df()
    _torder = ["tri_0", "tri_1_10", "tri_11_100", "tri_101_1000",
               "tri_1001_10000", "tri_10001p"]
    _tbar = alt.Chart(_tri).mark_bar().encode(
        x=alt.X("tri_class:N", sort=_torder, title="triangles per node (bins)"),
        y="n:Q").properties(
        width=330, height=220,
        title="Triangle heavy tail (median 161, max 30,025) — the clipping risk")
    _clo = con.execute(
        "select degree_class, closure_rate, wedges from mart_wedge_closure "
        "where closure_rate is not null").df()
    _order = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16",
              "deg_17_32", "deg_33_64", "deg_65p"]
    _cbar = alt.Chart(_clo).mark_bar().encode(
        x=alt.X("degree_class:N", sort=_order, title="wedge-center degree class"),
        y=alt.Y("closure_rate:Q", title="P(wedge closed)"),
        tooltip=["degree_class", "closure_rate", "wedges"]).properties(
        width=330, height=220,
        title="Wedge-closure rate by degree class (the sequence analog)")
    mo.ui.altair_chart(_tbar | _cbar)
    return


@app.cell
def _(alt, con, mo, pd):
    _sizes = con.execute(
        "select community_rank, n_nodes from mart_community_sizes "
        "order by n_nodes desc").df()
    _sbar = alt.Chart(_sizes).mark_bar().encode(
        x=alt.X("community_rank:N",
                sort=alt.EncodingSortField("n_nodes", order="descending"),
                title="community (rank label)"),
        y="n_nodes:Q").properties(width=330, height=240,
                                  title="Louvain community sizes (seed 7)")
    _mix = con.execute(
        "select pair_cell, n_edges from mart_community_mixing").df()
    _mix = pd.concat([_mix, _mix.pair_cell.str.split("__", expand=True)
                      .rename(columns={0: "ra", 1: "rb"})], axis=1)
    _rorder = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "rest"]
    _heat = alt.Chart(_mix).mark_rect().encode(
        x=alt.X("rb:N", sort=_rorder, title="rank b"),
        y=alt.Y("ra:N", sort=_rorder, title="rank a"),
        color=alt.Color("n_edges:Q",
                        scale=alt.Scale(type="symlog", scheme="blues")),
        tooltip=["pair_cell", "n_edges"]).properties(
        width=280, height=240,
        title="Community mixing (96.15% of edges on the diagonal)")
    mo.vstack([
        mo.md("## Community structure (rank labels only — the rank→member "
              "map never leaves)"),
        mo.ui.altair_chart(_sbar | _heat),
    ])
    return


@app.cell
def _(alt, con, mo):
    _cat = con.execute(
        "select category, n_nodes from mart_feature_categories "
        "order by n_nodes desc").df()
    _cbar = alt.Chart(_cat).mark_bar().encode(
        x=alt.X("n_nodes:Q", title="nodes with ≥1 feature in category"),
        y=alt.Y("category:N", sort="-x")).properties(
        width=330, height=240, title="Feature-category coverage (public taxonomy)")
    _circ = con.execute(
        "select circle_class, n_nodes from mart_circle_stats").df()
    _corder = ["circ_0", "circ_1", "circ_2", "circ_3", "circ_4p"]
    _cibar = alt.Chart(_circ).mark_bar().encode(
        x=alt.X("circle_class:N", sort=_corder, title="circle memberships"),
        y="n_nodes:Q").properties(width=330, height=240,
                                  title="Circle memberships per node")
    mo.ui.altair_chart(_cbar | _cibar)
    return


@app.cell
def _(con, mo):
    _m = con.execute(
        'select motif, n_nodes, round("share", 4) as "share" '
        "from mart_motif_summary order by n_nodes desc").df()
    mo.vstack([
        mo.md("## Motifs (public thresholds; SQL verdicts re-decided in Lean)\n"
              "hub = degree ≥ 150 · leaf = degree 1 · broker = degree ≥ 17 "
              "∧ cc < 1/5 (structural hole) · bridge = neighbors in ≥ 3 "
              "communities · butterfly = ≥ 3 circles in one ego · core = "
              "degree ≥ 65 ∧ cc ≥ 3/5 (broker ∧ core is **provably** "
              "impossible — `isBroker_not_isCore`). Node ids withheld."),
        mo.ui.table(_m, selection=None),
    ])
    return


@app.cell
def _(alt, con, mo):
    _assoc = con.execute(
        "select feat_a, feat_b, round(chi2,1) chi2, dof, "
        "round(cramers_v,3) cramers_v from mart_feature_assoc "
        "order by cramers_v desc limit 15").df()
    _bar = alt.Chart(_assoc).mark_bar().encode(
        x=alt.X("cramers_v:Q", title="Cramér's V"),
        y=alt.Y("pair:N", sort="-x", title=None),
        tooltip=["feat_a", "feat_b", "chi2", "dof", "cramers_v"]).transform_calculate(
        pair="datum.feat_a + ' × ' + datum.feat_b").properties(
        width=480, height=300, title="Top associations (χ²/Cramér's V)")
    mo.vstack([
        mo.md("## Feature associations (χ² recomputed exactly over ℚ in Lean)\n"
              "Co-occurrence between discretized node features, feature "
              "categories and motifs — full contingency grids live in "
              "`mart_assoc_cells` (enclave-only)."),
        mo.ui.altair_chart(_bar),
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
            "select * from mart_dp_degree").fetchall()}
        _rows = [{"bin": c["cell"], "value": int(c["noised_count"]),
                  "kind": "released (ε-DP)"} for c in _rel["cells"]]
        _rows += [{"bin": c, "value": n, "kind": "clipped (pre-noise, enclave)"}
                  for c, n in _clipped.items()]
        _order = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16",
                  "deg_17_32", "deg_33_64", "deg_65p"]
        _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
            x=alt.X("bin:N", sort=_order),
            y="value:Q", color="kind:N", xOffset="kind:N").properties(
            width=480, height=240,
            title="DP release utility: degree histogram, released vs pre-noise")
        _led = _rel["budget"]
        _out = mo.vstack([
            mo.md(f"## The ε-DP release (what actually leaves)\n"
                  f"Privacy unit: **one derived node row** (n = 4,039; the "
                  f"neighbor-induced node-DP gap is declared, not hidden — "
                  f"see goal.md). Heavy tails are clipped by proportional L1 "
                  f"at 8. Ledger: **{_led['spent_num']}/{_led['spent_den']}** "
                  f"of 4/1. The degree histogram has sensitivity 1 (each "
                  f"node in exactly one bin) — compare bar pairs to judge "
                  f"SNR honestly."),
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
        _agr = _res["agreement"]
        _out = mo.md(
            f"## Formal layer\n"
            f"Claims: **{_proved} proved**, {_assumed} assumed (trust boundary "
            f"stated explicitly). SQL↔Lean agreement: "
            f"**{_agr['n_agree']}/{_agr['n_checked']}** "
            f"({_res['motif_checks']} motif verdicts + {_res['class_checks']} "
            f"class assignments + {len(_res['assoc'])} exact χ² "
            f"recomputations), {_agr['n_disagree']} disagreements.")
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
        verified DP noise). The library `lean/EnclaveGraph/` holds the
        definitions and proofs; the repo-root `lean/*.lean` files are the
        executables `lake` links against it:

        | File | What it formalizes |
        |---|---|
        | `EnclaveGraph/Params.lean` | the concept spec mirrored from dbt `vars:` + the category taxonomy (locked by `tests/test_spec_sync.py`) |
        | `EnclaveGraph/Motifs.lean` | **decidable** motif predicates — `IsHub`, `IsLeaf`, `IsBroker` (degree ≥ 17 AND cc < 1/5 as the exact `Nat` cross-multiplication, no floats), `IsBridge`, `IsButterfly`, `IsCore` — with canonical-example, monotonicity and mutual-exclusion theorems (`isBroker_not_isCore`) |
        | `EnclaveGraph/GraphShape.lean` | the TOTAL public class maps (degree/clustering/triangle/richness/circle/community-size → grid cell, membership proved for every input), the 45-cell mixing and 16-cell closure grids (sizes kernel-checked against the ledger), GraphML rendering helpers |
        | `EnclaveGraph/Stats.lean` | clipped per-node cell sums; `clippedSum_sensitivity` (add/remove one row moves a cell by ≤ clip) |
        | `EnclaveGraph/DPRelease.lean` | the release mechanism on SampCert discrete Laplace: `privClippedCell_PureDP` (per cell) and `histMech_DP` (the composed K-cell histogram is K·ε₁/ε₂-DP) |
        | `EnclaveGraph/Sampler.lean` | the verified categorical sampler; `sampleCellPMF_law`: P(i) = wᵢ/Σw, exactly |
        | `EnclaveGraph/Assoc.lean` | χ² recomputed over ℚ; `chiSq_nonneg`, `chiSq_eq_zero_of_indep` |
        | `EnclaveGraph/GenSpec.lean` | the generator's typed spec + smart constructors with validity **by construction**: canonical duplicate-free edges (`mkEdge_ok`, `edgesCanonical_sound`, `edges_nodup`), duplicate-free category draws (`addPick_nodup`) |
        | `Release.lean` / `Checks.lean` / `Generate.lean` / `Sample.lean` / `Catsample.lean` | executables: ε-DP release (exact-fraction ledger, exit 2 over budget), SQL↔Lean re-deciding, the standalone synthetic GraphML generator, sampler tools |

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
        _files = (sorted((ROOT / "lean" / "EnclaveGraph").glob("*.lean"))
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
