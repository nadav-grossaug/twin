# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb", "networkx"]
# ///
"""Synth-vs-real comparison notebook: degree-class shares, clustering level,
community sizes, feature richness and categories, per-column TV scorecard
with each column's fill source, planted-motif recall and certificates.
Declared-gap framing throughout."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import networkx as nx
    import pandas as pd

    ROOT = Path(__file__).resolve().parent

    def _db():
        import duckdb
        try:
            return duckdb.connect(str(ROOT / "artifacts" / "enclave.duckdb"),
                                  read_only=True)
        except Exception:
            return None

    con = _db()
    synth_p = ROOT / "synth" / "synth_graph.graphml"
    sg = nx.read_graphml(synth_p) if synth_p.exists() else None
    plan_p = ROOT / "synth" / "plan.json"
    plan = json.loads(plan_p.read_text()) if plan_p.exists() else {}
    spec_p = ROOT / "lean" / "out" / "genspec.json"
    genspec = json.loads(spec_p.read_text()) if spec_p.exists() else {}

    CATEGORIES = ["birthday", "education", "first_name", "gender", "hometown",
                  "languages", "last_name", "locale", "location", "work",
                  "other"]

    def deg_bin(d):
        if d <= 1: return "deg_1"
        if d == 2: return "deg_2"
        if d <= 4: return "deg_3_4"
        if d <= 8: return "deg_5_8"
        if d <= 16: return "deg_9_16"
        if d <= 32: return "deg_17_32"
        if d <= 64: return "deg_33_64"
        return "deg_65p"

    def rich_bin(r):
        return f"rich_{r}" if r < 8 else "rich_8p"

    def csize_bin(s):
        if s < 25: return "csize_lt25"
        if s < 51: return "csize_25_50"
        if s < 101: return "csize_51_100"
        if s < 201: return "csize_101_200"
        if s < 401: return "csize_201_400"
        return "csize_gt400"

    def cat_of_key(k):
        # synthetic + real keys: 'ego_<e>__f<i>_<category...>_anonymized...'
        if "__f" not in k:
            return None
        tail = k.split("__f", 1)[1]
        tail = tail.split("_", 1)[1] if "_" in tail else tail
        for c in CATEGORIES:
            if tail.startswith(c):
                return c
        return "other"

    def node_cats(attrs):
        return {c for c in (cat_of_key(k) for k in attrs) if c}

    return (CATEGORIES, Counter, ROOT, alt, cat_of_key, con, csize_bin,
            deg_bin, genspec, json, mo, node_cats, nx, pd, plan, rich_bin, sg)


@app.cell
def _(mo, sg):
    mo.md(
        "# Synthetic vs real social graph — scorecard\n"
        "The generator targets the **spec contract and the DP release**, not "
        "realism; gaps are declared, measured and reported, never hidden."
        + ("" if sg is not None else "\n\n*(no synth yet — run `make synth`)*"))
    return


@app.cell
def _(Counter, alt, con, deg_bin, genspec, mo, pd, sg):
    _rows = []
    if con is not None:
        for _b, _n in con.execute(
                "select degree_class, n_nodes from mart_degree_stats").fetchall():
            _rows.append({"src": "real (enclave)", "bin": _b, "n": _n})
    for _c in genspec.get("degree", {}).get("cells", []):
        _rows.append({"src": "released (ε-DP weights)", "bin": _c["name"],
                      "n": _c["w"]})
    if sg is not None:
        _cnt = Counter(deg_bin(d) for _, d in sg.degree())
        for _b, _n in _cnt.items():
            _rows.append({"src": "synthetic", "bin": _b, "n": _n})
    _out = mo.md("*(run `make synth`)*")
    if _rows:
        _df = pd.DataFrame(_rows)
        _df["share"] = _df.n / _df.groupby("src").n.transform("sum")
        _order = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16",
                  "deg_17_32", "deg_33_64", "deg_65p"]
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("bin:N", sort=_order, title="public degree bins"),
            y=alt.Y("share:Q", title="share"),
            color="src:N", xOffset="src:N").properties(
            width=560, height=260,
            title="Degree-class shares: real vs released vs synthetic")
        _out = mo.vstack([
            mo.md("## Degree distribution (learning target 1)\n"
                  "The synth column should track the **released** weights — "
                  "that is its target; drift from *real* shares is the DP "
                  "noise plus clipping, and is honest."),
            mo.ui.altair_chart(_chart)])
    _out
    return


@app.cell
def _(con, genspec, mo, nx, pd, sg):
    _rows = []
    if con is not None:
        _s = con.execute("select * from mart_triangle_stats").df().iloc[0]
        _rows.append({"graph": "real (enclave)",
                      "avg_clustering": float(_s["avg_clustering"]),
                      "transitivity": float(_s["transitivity"]),
                      "nodes": int(_s["n_nodes"])})
    if sg is not None:
        _rows.append({"graph": "synthetic",
                      "avg_clustering": round(nx.average_clustering(sg), 4),
                      "transitivity": round(nx.transitivity(sg), 4),
                      "nodes": sg.number_of_nodes()})
    _released = None
    _cl, _op = 0, 0
    for _row in genspec.get("closure", {}).get("rows", []):
        for _c in _row.get("cells", []):
            if _c["name"] == "closed":
                _cl += _c["w"]
            elif _c["name"] == "open":
                _op += _c["w"]
    if _cl + _op > 0:
        _released = round(_cl / (_cl + _op), 4)
    _out = mo.md("*(run `make synth`)*")
    if _rows:
        _out = mo.vstack([
            mo.md("## Triadic closure (learning target 3)\n"
                  f"Released closure level (from the DP wedge grid): "
                  f"**{_released if _released is not None else '—'}** — the "
                  "generator's closure pass drives the synthetic graph toward "
                  "this level (validation band ±0.15)."),
            mo.ui.table(pd.DataFrame(_rows), selection=None)])
    _out
    return


@app.cell
def _(Counter, alt, con, csize_bin, mo, nx, pd, sg):
    _rows = []
    if con is not None:
        for (_n,) in con.execute(
                "select n_nodes from mart_community_sizes").fetchall():
            _rows.append({"src": "real (enclave)", "bin": csize_bin(_n)})
    if sg is not None:
        _comms = nx.community.louvain_communities(sg, seed=7)
        for _c in _comms:
            _rows.append({"src": "synthetic", "bin": csize_bin(len(_c))})
    _out = mo.md("*(run `make synth`)*")
    if _rows:
        _cnt = Counter((r["src"], r["bin"]) for r in _rows)
        _df = pd.DataFrame(
            [{"src": s, "bin": b, "n": n} for (s, b), n in _cnt.items()])
        _order = ["csize_lt25", "csize_25_50", "csize_51_100",
                  "csize_101_200", "csize_201_400", "csize_gt400"]
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("bin:N", sort=_order, title="community-size bins"),
            y=alt.Y("n:Q", title="communities"),
            color="src:N", xOffset="src:N").properties(
            width=520, height=240,
            title="Community sizes: real vs synthetic (both Louvain, seed 7)")
        _out = mo.vstack([
            mo.md("## Community structure (learning target 2)\n"
                  "Validation compares size-bin histograms, never raw labels "
                  "(Louvain on a synthetic graph may find a different k)."),
            mo.ui.altair_chart(_chart)])
    _out
    return


@app.cell
def _(mo, node_cats, nx, pd, plan, sg):
    _out = mo.md("*(run `make synth`)*")
    if sg is not None:
        _planted = plan.get("planted", {})
        _deg = dict(sg.degree())

        def _circles_max_one_ego(v):
            best = 0
            for k, val in sg.nodes[v].items():
                if k.endswith("__circles") and val:
                    best = max(best, len(str(val).split("|")))
            return best

        _n0, _n1, _n2 = "0", "1", "2"
        _cc2 = nx.clustering(sg, _n2) if _n2 in sg else None
        _rows = [
            {"motif": "hub", "node": 0,
             "planned": _planted.get("hub", {}).get("degree"),
             "observed": _deg.get(_n0),
             "recalled": bool(_deg.get(_n0, 0) >= 150)},
            {"motif": "broker", "node": 2,
             "planned": f"deg {_planted.get('broker', {}).get('degree')}, "
                        f"tri {_planted.get('broker', {}).get('triangles')}",
             "observed": f"deg {_deg.get(_n2)}, cc {_cc2:.3f}"
                         if _cc2 is not None else None,
             "recalled": bool(_deg.get(_n2, 0) >= 17 and _cc2 is not None
                              and _cc2 < 0.2)},
            {"motif": "butterfly", "node": 1,
             "planned": _planted.get("butterfly", {}).get("circles"),
             "observed": _circles_max_one_ego(_n1),
             "recalled": bool(_circles_max_one_ego(_n1) >= 3)},
        ]
        _nodes_with_cats = sum(1 for v in sg.nodes
                               if node_cats(sg.nodes[v]))
        _loops = list(nx.selfloop_edges(sg))
        _plan_edges = plan.get("n_edges")
        _plan_edges_s = f"{_plan_edges:,}" if isinstance(_plan_edges, int) else "—"
        _rows_df = pd.DataFrame(_rows)
        _out = mo.vstack([
            mo.md("## Planted-motif recall (ground truth from plan.json)\n"
                  "The generator SELF-VERIFIES each plant with the decidable "
                  "Lean predicates before writing the plan; the round-trip "
                  "re-detects them through the production pipeline."),
            mo.ui.table(_rows_df, selection=None),
            mo.md(f"**Edge canonicality**: {sg.number_of_nodes():,} nodes, "
                  f"{sg.number_of_edges():,} edges, "
                  f"{len(_loops)} self-loops (plan: "
                  f"{_plan_edges_s} canonical edges — the Lean "
                  f"checker `edgesCanonical` proved sorted/duplicate-free/"
                  f"loop-free before writing). "
                  f"{_nodes_with_cats:,} nodes carry ≥1 feature."),
        ])
    _out
    return


@app.cell
def _(CATEGORIES, Counter, alt, con, mo, node_cats, pd, rich_bin, sg):
    def _tv(a, b):
        ta, tb = sum(a.values()), sum(b.values())
        keys = set(a) | set(b)
        return (0.5 * sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb)
                          for k in keys) if ta and tb else None)

    _out = mo.md("*(run `make synth`)*")
    if sg is not None and con is not None:
        # real distributions from the warehouse
        _real_deg = dict(con.execute(
            "select degree_class, n_nodes from mart_degree_stats").fetchall())
        _real_rich = dict(con.execute(
            "select richness_class, count(*) from feat_nodes group by 1"
        ).fetchall())
        _real_cat = dict(con.execute(
            "select category, n_nodes from mart_feature_categories").fetchall())
        _real_circ = dict(con.execute(
            "select circle_class, n_nodes from mart_circle_stats").fetchall())
        # synth distributions from the graphml
        from collections import Counter as _C

        def _deg_bin(d):
            if d <= 1: return "deg_1"
            if d == 2: return "deg_2"
            if d <= 4: return "deg_3_4"
            if d <= 8: return "deg_5_8"
            if d <= 16: return "deg_9_16"
            if d <= 32: return "deg_17_32"
            if d <= 64: return "deg_33_64"
            return "deg_65p"

        _syn_deg = _C(_deg_bin(d) for _, d in sg.degree())
        _syn_rich = _C(rich_bin(len(node_cats(sg.nodes[v]))) for v in sg.nodes)
        _syn_cat = _C()
        for v in sg.nodes:
            for c in node_cats(sg.nodes[v]):
                _syn_cat[c] += 1

        def _circ_bin(c):
            return f"circ_{c}" if c < 4 else "circ_4p"

        def _n_circ(v):
            n = 0
            for k, val in sg.nodes[v].items():
                if k.endswith("__circles") and val:
                    n += len(str(val).split("|"))
            return n

        _syn_circ = _C(_circ_bin(_n_circ(v)) for v in sg.nodes)
        _score = [
            {"column": "degree class", "tv_real_vs_synth": _tv(_real_deg, _syn_deg),
             "fill_source": "DP degree histogram + mixing rows + closure pass"},
            {"column": "feature richness", "tv_real_vs_synth": _tv(_real_rich, _syn_rich),
             "fill_source": "DP richness histogram"},
            {"column": "category coverage", "tv_real_vs_synth": _tv(_real_cat, _syn_cat),
             "fill_source": "DP category-presence histogram"},
            {"column": "circle memberships", "tv_real_vs_synth": _tv(_real_circ, _syn_circ),
             "fill_source": "DP circle histogram"},
        ]
        _df = pd.DataFrame([{**r, "tv_real_vs_synth":
                             None if r["tv_real_vs_synth"] is None
                             else round(r["tv_real_vs_synth"], 4)}
                            for r in _score])
        _chart = alt.Chart(_df).mark_bar().encode(
            x=alt.X("tv_real_vs_synth:Q", scale=alt.Scale(domain=[0, 1])),
            y="column:N", color=alt.Color("fill_source:N"),
            tooltip=["column", "tv_real_vs_synth", "fill_source"]).properties(
            width=520, height=160, title="Per-column TV, real vs synth")
        _out = mo.vstack([mo.md(
            "## TV scorecard\n"
            "**TV = total variation distance** between the real and synthetic "
            "distributions over a column's public bins: "
            "TV = ½ · Σ|p_real(bin) − p_synth(bin)|, ranging **0 to 1** — "
            "0 means identical mixes, 1 means completely disjoint. Read it as "
            "*the share of probability mass that would have to move to make "
            "the two distributions match*. The generator targets the DP "
            "release, so some drift from the *real* data is expected and "
            "honest."),
            mo.ui.altair_chart(_chart),
            mo.ui.table(_df, selection=None)])
    _out
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
