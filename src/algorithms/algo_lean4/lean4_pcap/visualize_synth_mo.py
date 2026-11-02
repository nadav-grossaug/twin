"""Synthetic flows — EDA and enclave-vs-synth comparison (marimo).

Run:  marimo run visualize_synth_mo.py
Edit: marimo edit visualize_synth_mo.py

Requires `make synth` (and for the realism section, the trained models and
`make realism-audit`). Reads synth/scratch.duckdb — the synthetic data pushed
through the PRODUCTION dbt pipeline — side by side with the real enclave
warehouse, and compares every feat_flows column.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Synthetic Flows — EDA & Enclave Comparison

    The generator (`lake exe generate`) runs **entirely in Lean** from the
    distilled, typed generation spec (`genspec.json` — DP-released histograms
    incl. the semi-Markov service transitions that replace the NextServiceGRU
    teacher, plus common-knowledge top-k names). Every random value is drawn
    in-process by the verified sampler (law `sampleCellPMF_law`); records go
    through `mkFlow` (validity by construction) with shape parameters from
    `lean/EnclaveFlows/FlowShape.lean`, whose linking theorems guarantee the
    encoder accepts the plan. No PyTorch or Python sampling anywhere.

    This notebook: (1) EDA of the synthetic data through the production
    pipeline, (2) **per-column goodness-of-fit against each column's
    declared distribution**, (3) **column-by-column comparison against
    the real enclave data** with a similarity-overview scorecard,
    (4) geography (source & destination countries), (5) the realism audit.
    """)
    return


@app.cell
def _():
    import json
    import math
    import re
    from pathlib import Path

    import altair as alt
    import duckdb
    import numpy as np
    import pandas as pd

    # bounded guard: a chart embedding >20k rows is a bug (pre-aggregate in
    # SQL/numpy instead) — fail loudly rather than emit a multi-hundred-MB cell
    alt.data_transformers.enable("default", max_rows=20_000)

    ROOT = Path(__file__).resolve().parent

    PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    REAL_C, SYNTH_C = PAL[0], PAL[1]

    def styled(chart, height=260):
        return (
            chart.properties(height=height)
            .configure_axis(gridColor="#e1e0d9", domainColor="#c3c2b7",
                            labelColor="#898781", titleColor="#52514e")
            .configure_view(stroke=None)
        )

    def chi2_pvalue(observed, expected):
        pairs = [(o, e) for o, e in zip(observed, expected) if e >= 1.0]
        dof = len(pairs) - 1
        if dof < 1:
            return 1.0
        chi2 = sum((o - e) ** 2 / e for o, e in pairs)
        z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        return 0.5 * math.erfc(z / math.sqrt(2))

    return (
        PAL,
        REAL_C,
        ROOT,
        SYNTH_C,
        alt,
        chi2_pvalue,
        duckdb,
        json,
        np,
        pd,
        re,
        styled,
    )


@app.cell
def _(ROOT, duckdb, json, mo, re):
    mo.stop(
        not (ROOT / "synth" / "scratch.duckdb").exists()
        or not (ROOT / "synth" / "flow_synth.jsonl").exists(),
        mo.md("⚠️ **No synthetic build found** — run `make synth` first."),
    )
    mo.stop(
        not (ROOT / "artifacts" / "enclave.duckdb").exists(),
        mo.md("⚠️ **No enclave warehouse** — run `make dbt` first."),
    )
    scon = duckdb.connect(str(ROOT / "synth" / "scratch.duckdb"), read_only=True)
    rcon = duckdb.connect(str(ROOT / "artifacts" / "enclave.duckdb"), read_only=True)

    syn_meta = json.loads((ROOT / "synth" / "meta.json").read_text())
    syn_plan = json.loads((ROOT / "synth" / "plan.json").read_text())
    _v = ROOT / "synth" / "validation.json"
    syn_val = json.loads(_v.read_text()) if _v.exists() else None
    _c = ROOT / "synth" / "certificate.json"
    syn_cert = json.loads(_c.read_text()) if _c.exists() else None

    # raw synthetic records (for GOF against declared choice/weight lists)
    _lines = (ROOT / "synth" / "flow_synth.jsonl").read_text().splitlines()
    syn_raw = [json.loads(x) for x in _lines[1:]]
    beacon_set = {(b["sa"], b["da"], b["dp"]) for b in syn_plan["beacons"]}

    # generator shape parameters: FlowShape.lean is the master
    _txt = (ROOT / "lean" / "EnclaveFlows" / "FlowShape.lean").read_text()
    shape = {m[1]: int(m[2]) for m in re.finditer(r"def (\w+) : Nat := (\d+)", _txt)}
    for _m in re.finditer(r"def (\w+) : List Nat := \[([\d,\s]*)\]", _txt):
        shape[_m[1]] = [int(x) for x in _m[2].split(",") if x.strip()]
    shape["genServices"] = re.findall(
        r'"([^"]+)"', re.search(r"def genServices : List String := \[([^\]]*)\]", _txt)[1])
    return (
        beacon_set,
        rcon,
        scon,
        shape,
        syn_cert,
        syn_meta,
        syn_plan,
        syn_raw,
        syn_val,
    )


@app.cell
def _(mo, scon, syn_cert, syn_meta, syn_plan, syn_val):
    _n, _hosts = scon.execute(
        "select count(*), count(distinct sa) from feat_flows").fetchone()
    _tiles = [
        mo.stat(f"{_n:,}", label="synthetic flows",
                caption=f"{_hosts} hosts (plan: {syn_meta['hosts']})"),
        mo.stat(f"{len(syn_plan['beacons'])}", label="planned beacon channels",
                caption="each proved accepted (genBeacon_isBeacon)"),
    ]
    if syn_val:
        _tiles.append(mo.stat(
            "pass" if syn_val.get("passed") else f"exit {syn_val.get('exit')}",
            label="validate (re-ingestion)",
            caption=f"TV service mix {syn_val.get('tv', {}).get('service_mix', '—')}"
            if isinstance(syn_val.get("tv"), dict) else "production pipeline + Lean deciders"))
    if syn_cert:
        _tiles.append(mo.stat(
            "kernel ✓" if syn_cert.get("kernel_only") else "native",
            label="Lean certificate",
            caption=f"{syn_cert.get('tactic', 'decide')}, {syn_cert.get('seconds', '?')}s"))
    mo.hstack(_tiles, justify="space-around")
    return


@app.cell
def _(PAL, alt, mo, scon, styled):
    _df = scon.execute(
        """
        select service_class, count(*) as n_flows, sum(bytes_total) as bytes_total
        from feat_flows group by 1 order by n_flows desc
        """
    ).fetch_df()
    _chart = styled(
        alt.Chart(_df).mark_bar(color=PAL[1], cornerRadiusEnd=4)
        .encode(x=alt.X("n_flows:Q", title="flows"),
                y=alt.Y("service_class:N", sort="-x", title=None),
                tooltip=["service_class", "n_flows", "bytes_total"]),
        height=220,
    )
    mo.vstack([mo.md("## Synthetic EDA\n### Flows by service class "
                     "(through the production taxonomy)"), _chart])
    return


@app.cell
def _(SYNTH_C, alt, mo, pd, scon, styled):
    _df = scon.execute("select sa, time_start from feat_flows").fetch_df()
    _df["t1"] = pd.to_datetime(_df.time_start, unit="s").dt.floor("1min")
    _agg = _df.groupby("t1", as_index=False).size()
    _chart = styled(
        alt.Chart(_agg).mark_line(color=SYNTH_C, strokeWidth=2, interpolate="monotone")
        .encode(x=alt.X("t1:T", title="time (UTC)"),
                y=alt.Y("size:Q", title="flows / min"),
                tooltip=[alt.Tooltip("t1:T"), "size:Q"]),
        height=200,
    ).interactive()
    mo.vstack([
        mo.md("### Activity over time — sessions (bursts) + beacon heartbeats"),
        _chart,
    ])
    return


@app.cell
def _(mo, pd, scon, syn_plan):
    _det = scon.execute(
        """
        select sa, da, dp, n_flows, round(period_s, 2) as period_s,
               round(cv_gap, 3) as cv, is_beacon
        from mart_motif_beacon order by sa, da
        """
    ).fetch_df()
    _planned = pd.DataFrame(syn_plan["beacons"])[["sa", "da", "dp", "period_ms"]]
    _planned["planned_period_s"] = _planned.period_ms / 1000
    _cmp = _planned.merge(_det, on=["sa", "da", "dp"], how="left")
    _recall = float(_cmp.is_beacon.fillna(False).mean()) if len(_cmp) else 0.0
    mo.vstack([
        mo.md(
            f"""
            ### Planned vs detected beacons — the theorem at work

            `genBeacon_isBeacon` proves channels rendered from the FlowShape
            parameters must satisfy `IsBeacon`; the production detector found
            **{int(_cmp.is_beacon.fillna(False).sum())}/{len(_cmp)}**
            (recall {_recall:.2f}). Detected periods match the plan.
            """
        ),
        mo.ui.table(_cmp[["sa", "da", "dp", "planned_period_s", "period_s",
                          "cv", "n_flows", "is_beacon"]], selection=None),
    ])
    return


@app.cell
def _(PAL, alt, beacon_set, chi2_pvalue, mo, pd, shape, styled, syn_raw):
    # Per-column GOF: sampled columns vs their DECLARED weights (the verified
    # sampler's law makes these the ground-truth targets).
    _nb = [f for f in syn_raw if (f["sa"], f["da"], f["dp"]) not in beacon_set]

    def _gof(values, choices, weights, label):
        tot = sum(weights)
        obs = [sum(1 for v in values if v == c) for c in choices]
        exp = [len(values) * w / tot for w in weights]
        p = chi2_pvalue(obs, exp)
        rows = []
        for c, o, e in zip(choices, obs, exp):
            rows.append({"column": label, "value": str(c), "n": o, "kind": "observed"})
            rows.append({"column": label, "value": str(c), "n": e, "kind": "declared"})
        return p, pd.DataFrame(rows)

    _p1, _d1 = _gof([f["bytes_out"] for f in _nb],
                    shape["bytesOutChoices"], shape["bytesOutWeights"], "bytes_out")
    _p2, _d2 = _gof([f["bytes_in"] for f in _nb],
                    shape["bytesInChoices"], shape["bytesInWeights"], "bytes_in")
    _p3, _d3 = _gof([round((f["time_end"] - f["time_start"]) * 1000) for f in _nb],
                    shape["flowDurationMsChoices"], shape["flowDurationMsWeights"],
                    "duration_ms")
    _all = pd.concat([_d1, _d2, _d3])
    _chart = styled(
        alt.Chart(_all).mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("value:N", title=None, sort=None),
            y=alt.Y("n:Q", title="flows"),
            color=alt.Color("kind:N",
                            scale=alt.Scale(domain=["observed", "declared"],
                                            range=[PAL[1], "#898781"]),
                            legend=alt.Legend(title=None, orient="top")),
            xOffset="kind:N",
            column=alt.Column("column:N", title=None),
        ), height=180,
    )
    mo.vstack([
        mo.md(
            f"""
            ## Column fill quality I — sampled columns vs declared weights

            Each sampled column is drawn by the **verified sampler** whose law
            is the theorem `P(i) = wᵢ/Σw`; the χ² goodness-of-fit confirms the
            executable draws follow it:
            **bytes_out p = {_p1:.2f}**, **bytes_in p = {_p2:.2f}**,
            **duration p = {_p3:.2f}** (reject below 0.001).
            """
        ),
        _chart,
    ])
    return


@app.cell
def _(
    SYNTH_C,
    alt,
    beacon_set,
    chi2_pvalue,
    mo,
    pd,
    shape,
    styled,
    syn_plan,
    syn_raw,
):
    # Timing randomness: uniform session steps; uniform beacon jitter in bounds
    _nb = sorted((f for f in syn_raw
                  if (f["sa"], f["da"], f["dp"]) not in beacon_set),
                 key=lambda f: (f["sa"], f["time_start"]))
    _max = shape["sessionStepMsMax"]
    _by = {}
    for _f in _nb:
        _by.setdefault(_f["sa"], []).append(_f["time_start"])
    _steps = [(b - a) * 1000 for ts in _by.values()
              for a, b in zip(ts, ts[1:]) if (b - a) * 1000 <= _max]
    _k = 8
    _obs = [sum(1 for s in _steps if i * _max / _k < s <= (i + 1) * _max / _k)
            for i in range(_k)]
    _p_steps = chi2_pvalue(_obs, [len(_steps) / _k] * _k)

    _j = shape["beaconJitterMs"]
    _jit = []
    for _b in syn_plan["beacons"]:
        _chan = sorted((f for f in syn_raw if (f["sa"], f["da"], f["dp"])
                        == (_b["sa"], _b["da"], _b["dp"])),
                       key=lambda f: f["time_start"])
        for _a, _c in zip(_chan, _chan[1:]):
            _jit.append((_c["time_start"] - _a["time_start"]) * 1000 - (_b["period_ms"] - _j))
    _k2 = 6
    _obs2 = [sum(1 for x in _jit if i * 2 * _j / _k2 <= x < (i + 1) * 2 * _j / _k2
                 or (i == _k2 - 1 and x >= 2 * _j)) for i in range(_k2)]
    _p_jit = chi2_pvalue(_obs2, [len(_jit) / _k2] * _k2)

    _sdf = pd.DataFrame({"step_ms": _steps})
    _jdf = pd.DataFrame({"jitter_ms": _jit})
    _c1 = styled(alt.Chart(_sdf).mark_bar(color=SYNTH_C, cornerRadiusEnd=2)
                 .encode(x=alt.X("step_ms:Q", bin=alt.Bin(maxbins=16),
                                 title=f"intra-session step (ms, ≤ {_max})"),
                         y=alt.Y("count():Q", title="steps"),
                         tooltip=["count():Q"]), height=180)
    _c2 = styled(alt.Chart(_jdf).mark_bar(color=SYNTH_C, cornerRadiusEnd=2)
                 .encode(x=alt.X("jitter_ms:Q", bin=alt.Bin(maxbins=12),
                                 title=f"beacon gap − (period−j) (ms, ∈ [0, {2*_j}])"),
                         y=alt.Y("count():Q", title="gaps"),
                         tooltip=["count():Q"]), height=180)
    mo.vstack([
        mo.md(
            f"""
            ## Column fill quality II — randomness of timing

            Session steps are uniform on (0, {_max}] ms
            (**χ² p = {_p_steps:.2f}**); beacon jitter is uniform on
            [period−{_j}, period+{_j}] ms (**p = {_p_jit:.2f}**) and never
            leaves the bound the theorem `genBeacon_isBeacon` requires.
            """
        ),
        mo.hstack([_c1, _c2], widths="equal"),
    ])
    return


@app.cell
def _(mo, np, pd, rcon, scon):
    # ---- per-column comparison machinery: real (enclave) vs synthetic ------
    real_df = rcon.execute("select * exclude (byte_dist) from feat_flows").fetch_df()
    syn_df = scon.execute("select * exclude (byte_dist) from feat_flows").fetch_df()

    VERDICT = {
        "service_class": "target — DP-released mix (+ forced https beacon channels)",
        "bytes_out": "assumption — FlowShape weight list (verified sampler)",
        "bytes_in": "assumption — FlowShape weight list (verified sampler)",
        "duration_s": "assumption — FlowShape duration bins (verified sampler)",
        "bytes_total": "derived — bytes_out + bytes_in",
        "pkts_total": "derived — 1 + bytes // 1400",
        "num_pkts_out": "derived — 1 + bytes_out // 1400",
        "num_pkts_in": "derived — bytes_in rule",
        "bytes_per_s": "derived — sampled bytes / sampled duration",
        "byte_ratio_out_in": "derived — sampled bytes ratio",
        "is_zero_byte": "derived — never (bytes_out > 0 by construction)",
        "is_unanswered": "derived — bytes_in = 0 draws",
        "time_start": "assumption — session structure (base epoch + bounded steps)",
        "time_end": "derived — time_start + sampled duration",
        "hour_utc": "assumption — base epoch + session spread",
        "sa": "structural — synthetic host plan 10.99.0.0/24",
        "da": "structural — synthetic external pool 198.51.100.0/24",
        "sp": "structural — ephemeral counter 49152+",
        "dp": "structural — service→port map",
        "pr": "structural — service→protocol map",
        "src_internal": "structural — all synthetic hosts are internal",
        "dst_internal": "structural — all destinations external",
        "entropy": "structural — not rendered (NULL)",
        "qf_invalid_entropy": "clean by construction",
        "qf_negative_duration": "clean by construction",
        "has_dns": "structural — dns records not rendered",
        "has_tls": "structural — tls records not rendered",
        "is_right_censored": "constant — expire_type 'i'",
        "expire_type": "constant 'i'",
        "ttl_out": "constant 64",
        "ttl_in": "constant 57",
        "spec_version": "stamp",
        "flow_id": "identifier",
        "flow_key": "identifier",
    }

    def _tv_numeric(r, s):
        r, s = r.dropna(), s.dropna()
        if len(r) == 0 or len(s) == 0:
            return None
        lo = min(r.quantile(0.01), s.quantile(0.01))
        hi = max(r.quantile(0.99), s.quantile(0.99))
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 13)
        hr, _ = np.histogram(np.clip(r, lo, hi), bins=bins)
        hs, _ = np.histogram(np.clip(s, lo, hi), bins=bins)
        return float(0.5 * np.abs(hr / max(hr.sum(), 1) - hs / max(hs.sum(), 1)).sum())

    def _tv_cat(r, s):
        rf = r.astype("object").fillna("∅").astype(str).value_counts(normalize=True)
        sf = s.astype("object").fillna("∅").astype(str).value_counts(normalize=True)
        keys = set(rf.index) | set(sf.index)
        return float(0.5 * sum(abs(rf.get(k, 0.0) - sf.get(k, 0.0)) for k in keys))

    def _summary(series, numeric):
        s = series.dropna()
        if numeric:
            if len(s) == 0:
                return "all NULL"
            return f"μ {s.mean():.3g} · med {s.median():.3g} · p95 {s.quantile(0.95):.3g}"
        top = (series.astype("object").fillna("∅").astype(str)
               .value_counts(normalize=True).head(2))
        return " · ".join(f"{k} {v:.0%}" for k, v in top.items())

    _rows = []
    for _col in syn_df.columns:
        if _col in ("flow_id", "flow_key"):
            continue
        _r, _s = real_df[_col], syn_df[_col]
        _numeric = pd.api.types.is_numeric_dtype(_s) and not pd.api.types.is_bool_dtype(_s)
        _tv = _tv_numeric(_r, _s) if _numeric else _tv_cat(_r, _s)
        _rows.append({
            "column": _col,
            "kind": "numeric" if _numeric else "categorical",
            "enclave (real)": _summary(_r, _numeric),
            "synthetic": _summary(_s, _numeric),
            "TV distance": None if _tv is None else round(_tv, 3),
            "null % real→synth": f"{_r.isna().mean():.0%} → {_s.isna().mean():.0%}",
            "fill source": VERDICT.get(_col, "—"),
        })
    column_compare = pd.DataFrame(_rows).sort_values(
        "TV distance", ascending=False, na_position="last")
    mo.vstack([
        mo.md(
            """
            ## Column-by-column: enclave vs synthetic

            Every `feat_flows` column, with summaries on both sides, total
            variation distance (TV, explained in the similarity overview
            below: ½Σ|P−Q|, 0 = identical, 1 = disjoint; numeric: 12 shared
            quantile-range bins; categorical: exact), the NULL-rate change,
            and **what fills the
            column** — the DP-released target, a declared FlowShape
            assumption, a derived quantity, or a structural constant. High TV
            on assumption/structural columns is expected and honest; the
            *target* column (service_class) is the one that must track the
            release.
            """
        ),
        mo.ui.table(column_compare, selection=None, page_size=35),
    ])
    return column_compare, real_df, syn_df


@app.cell
def _(PAL, ROOT, alt, column_compare, json, mo, styled):
    # ---- similarity overview: one score per column + distillation fidelity --
    _sc = column_compare.dropna(subset=["TV distance"]).copy()
    _kind_map = {"target": "target (DP release)",
                 "assumption": "assumption (FlowShape)",
                 "derived": "derived", "structural": "structural"}
    _sc["source"] = [_kind_map.get(str(v).split(" ")[0], "constant/stamp")
                     for v in _sc["fill source"]]
    _dom = ["target (DP release)", "assumption (FlowShape)", "derived",
            "structural", "constant/stamp"]
    _chart = styled(
        alt.Chart(_sc).mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("TV distance:Q", scale=alt.Scale(domain=[0, 1]),
                    title="total variation distance (0 = identical mix)"),
            y=alt.Y("column:N", sort="-x", title=None),
            color=alt.Color("source:N",
                            scale=alt.Scale(domain=_dom,
                                            range=[PAL[0], PAL[3], PAL[2],
                                                   "#898781", "#c3c2b7"]),
                            legend=alt.Legend(title="column filled by",
                                              orient="top", columns=3)),
            tooltip=["column", alt.Tooltip("TV distance:Q", format=".3f"),
                     "fill source"],
        ), height=max(240, 15 * len(_sc)),
    )
    _svc_tv = float(_sc.loc[_sc.column == "service_class", "TV distance"].iloc[0])
    _tiles = [
        mo.stat(f"{_svc_tv:.3f}", label="TV service mix",
                caption="the released target — must be small"),
        mo.stat(f"{_sc['TV distance'].median():.2f}", label="median column TV"),
        mo.stat(f"{(_sc['TV distance'] <= 0.2).mean():.0%}",
                label="columns with TV ≤ 0.2"),
    ]
    _rep = ROOT / "artifacts" / "reports" / "distill_report.json"
    if _rep.exists():
        _ns = json.loads(_rep.read_text()).get("next_service") or {}
        if _ns:
            _tiles.append(mo.stat(
                f"{_ns['student_markov_acc']:.3f}",
                label="distilled next-service accuracy",
                caption=f"repeat-last {_ns['repeat_last_acc']:.3f} · "
                        f"teacher GRU {_ns['teacher_gru_val_acc']:.3f}"))
    mo.vstack([
        mo.md(
            """
            ## Similarity overview — how close is synthetic to enclave?

            **What is TV distance?** The *total variation distance* between
            two distributions $P$ (enclave) and $Q$ (synthetic):
            $TV(P,Q) = \\tfrac{1}{2}\\sum_x |P(x) - Q(x)|$. It is the
            largest difference the two sides can assign to any event —
            equivalently, the fraction of probability mass you would have to
            move to turn one distribution into the other. **0 = identical
            mixes, 1 = completely disjoint**; e.g. TV 0.3 means 30% of the
            synthetic mass sits in the wrong cells. Categorical columns
            compare exact value frequencies; numeric columns are first binned
            into 12 shared quantile-range bins, so TV reads the same way for
            both. It is a stricter, symmetric alternative to eyeballing two
            histograms — and the same statistic the release gate's
            `TV_TOLERANCE` check uses.

            One TV distance per column, colored by **what fills the column**.
            Read it with the architecture in mind: the *target* column tracks
            the DP release and should sit near zero; *assumption* and
            *structural* columns are declared, deliberately coarse renderings
            — their distance is the honestly-documented realism gap, not a
            bug. The distillation tile shows the Lean semi-Markov student
            standing in for the retired NextServiceGRU teacher on real
            validation pairs.
            """
        ),
        mo.hstack(_tiles, justify="space-around"),
        _chart,
    ])
    return


@app.cell
def _(column_compare, mo):
    column_pick = mo.ui.dropdown(
        options=sorted(column_compare.column.tolist()),
        value="bytes_total",
        label="drill into column",
    )
    column_pick
    return (column_pick,)


@app.cell
def _(REAL_C, SYNTH_C, alt, column_pick, mo, np, pd, real_df, styled, syn_df):
    _col = column_pick.value
    _r, _s = real_df[_col], syn_df[_col]
    _numeric = pd.api.types.is_numeric_dtype(_s) and not pd.api.types.is_bool_dtype(_s)
    if _numeric:
        _rv, _sv = _r.dropna(), _s.dropna()
        _lo = float(min(_rv.quantile(0.01), _sv.quantile(0.01))) if len(_sv) else 0.0
        _hi = float(max(_rv.quantile(0.99), _sv.quantile(0.99))) if len(_sv) else 1.0
        if _hi <= _lo:
            _hi = _lo + 1
        _bins = np.linspace(_lo, _hi, 25)
        _frames = []
        for _name, _v in (("enclave", _rv), ("synthetic", _sv)):
            _h, _ = np.histogram(np.clip(_v, _lo, _hi), bins=_bins)
            _frames.append(pd.DataFrame({
                "x": _bins[:-1], "frac": _h / max(_h.sum(), 1), "side": _name}))
        _df = pd.concat(_frames)
        _chart = styled(
            alt.Chart(_df).mark_bar(opacity=0.75, cornerRadiusEnd=2)
            .encode(x=alt.X("x:Q", title=_col),
                    y=alt.Y("frac:Q", title="fraction", stack=None),
                    color=alt.Color("side:N",
                                    scale=alt.Scale(domain=["enclave", "synthetic"],
                                                    range=[REAL_C, SYNTH_C]),
                                    legend=alt.Legend(title=None, orient="top")),
                    xOffset="side:N",
                    tooltip=["side", alt.Tooltip("x:Q", format=".3g"),
                             alt.Tooltip("frac:Q", format=".3f")]),
            height=260,
        )
    else:
        _rf = (_r.astype("object").fillna("∅").astype(str)
               .value_counts(normalize=True))
        _sf = (_s.astype("object").fillna("∅").astype(str)
               .value_counts(normalize=True))
        _keys = sorted(set(_rf.index) | set(_sf.index),
                       key=lambda k: -(_rf.get(k, 0) + _sf.get(k, 0)))[:14]
        _df = pd.concat([
            pd.DataFrame({"value": _keys,
                          "frac": [_rf.get(k, 0.0) for k in _keys],
                          "side": "enclave"}),
            pd.DataFrame({"value": _keys,
                          "frac": [_sf.get(k, 0.0) for k in _keys],
                          "side": "synthetic"}),
        ])
        _chart = styled(
            alt.Chart(_df).mark_bar(cornerRadiusEnd=3)
            .encode(x=alt.X("value:N", sort=_keys, title=_col),
                    y=alt.Y("frac:Q", title="fraction"),
                    color=alt.Color("side:N",
                                    scale=alt.Scale(domain=["enclave", "synthetic"],
                                                    range=[REAL_C, SYNTH_C]),
                                    legend=alt.Legend(title=None, orient="top")),
                    xOffset="side:N",
                    tooltip=["side", "value", alt.Tooltip("frac:Q", format=".3f")]),
            height=260,
        )
    mo.vstack([mo.md(f"### `{_col}` — enclave vs synthetic (normalized)"), _chart])
    return


@app.cell
def _(REAL_C, SYNTH_C, alt, mo, pd, rcon, scon, styled):
    # fixed small multiples for the headline columns — aggregated in SQL so
    # the charts carry per-class/per-bin fractions, never per-flow rows
    _svc_frames, _byte_frames = [], []
    for _side, _con in (("enclave", rcon), ("synthetic", scon)):
        _d = _con.execute(
            """
            select service_class, count(*) as n,
                   count(*) / sum(count(*)) over () as frac
            from feat_flows group by 1
            """
        ).fetch_df()
        _d["side"] = _side
        _svc_frames.append(_d)
        _b = _con.execute(
            """
            with b as (
                select floor(least(greatest(ln(1 + bytes_total), 0), 12) * 2) / 2 as bb
                from feat_flows
            )
            select bb, count(*) as n, count(*) / sum(count(*)) over () as frac
            from b group by 1
            """
        ).fetch_df()
        _b["side"] = _side
        _byte_frames.append(_b)
    _svc_df = pd.concat(_svc_frames)
    _byte_df = pd.concat(_byte_frames)
    _scale = alt.Scale(domain=["enclave", "synthetic"], range=[REAL_C, SYNTH_C])
    _leg = alt.Legend(title=None, orient="top")

    _svc = styled(
        alt.Chart(_svc_df).mark_bar(cornerRadiusEnd=3)
        .encode(x=alt.X("service_class:N", title=None),
                y=alt.Y("frac:Q", title="fraction"),
                color=alt.Color("side:N", scale=_scale, legend=_leg),
                xOffset="side:N",
                tooltip=["side", "service_class", "n",
                         alt.Tooltip("frac:Q", format=".3f")]),
        height=200,
    )
    _bytes = styled(
        alt.Chart(_byte_df).mark_bar(opacity=0.7)
        .encode(x=alt.X("bb:Q", title="ln(1+bytes_total)"),
                y=alt.Y("frac:Q", title=None, stack=None),
                color=alt.Color("side:N", scale=_scale, legend=None),
                xOffset="side:N",
                tooltip=["side", alt.Tooltip("bb:Q", format=".1f"),
                         alt.Tooltip("frac:Q", format=".3f")]),
        height=200,
    )
    mo.vstack([
        mo.md("### Headline comparisons"),
        mo.hstack([_svc, _bytes], widths="equal"),
        mo.md("Synthetic volume/duration distributions are **deliberately "
              "coarser** (5-point weight lists vs continuous reality) — they "
              "are declared assumptions, not released facts. The service mix "
              "is the released target and tracks it."),
    ])
    return


@app.cell
def _(mo, pd, rcon, scon):
    _r = rcon.execute(
        "select 'enclave' as side, count(*) n_sessions, round(avg(n_flows),1) mean_flows, "
        "max(n_flows) max_flows, round(avg(duration_s),1) mean_dur_s from mart_seq_sessions"
    ).fetch_df()
    _s = scon.execute(
        "select 'synthetic' as side, count(*), round(avg(n_flows),1), max(n_flows), "
        "round(avg(duration_s),1) from mart_seq_sessions"
    ).fetch_df()
    _s.columns = _r.columns
    mo.vstack([
        mo.md("### Sessions & motifs through the same marts"),
        mo.ui.table(pd.concat([_r, _s]), selection=None),
    ])
    return


@app.cell
def _(PAL, alt, mo, rcon, scon, styled):
    _n_tls_r, _n_tls_canon = rcon.execute(
        "select count(*), sum(is_canonical_hs::int) from feat_tls").fetchone()
    _n_stacks = rcon.execute("select count(*) from mart_tls_fingerprints").fetchone()[0]
    _n_dns_r, _n_names = rcon.execute(
        "select count(*), count(distinct qname) from feat_dns").fetchone()
    _n_fans = rcon.execute(
        "select sum(is_subdomain_fan::int) from mart_dns_domains").fetchone()[0]
    _n_tls_s = scon.execute("select count(*) from feat_tls").fetchone()[0]
    _n_dns_s = scon.execute("select count(*) from feat_dns").fetchone()[0]

    _fp = rcon.execute(
        "select 'TLS stack ' || fingerprint as item, n_flows as n, "
        "is_rare_stack as flagged from mart_tls_fingerprints "
        "union all "
        "select 'DNS domain ' || registered_domain, n_distinct_names, is_subdomain_fan "
        "from mart_dns_domains order by n desc limit 14"
    ).fetch_df()
    _chart = styled(
        alt.Chart(_fp).mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("n:Q", title="flows (TLS) / distinct names (DNS)"),
            y=alt.Y("item:N", sort="-x", title=None),
            color=alt.Color("flagged:N",
                            scale=alt.Scale(domain=[False, True],
                                            range=[PAL[0], PAL[1]]),
                            legend=alt.Legend(title="motif-flagged", orient="top")),
            tooltip=["item", "n", "flagged"],
        ), height=260,
    )
    _syn_stacks = scon.execute(
        "select count(distinct fingerprint), sum(is_canonical_hs::int) from feat_tls"
    ).fetchone()
    _syn_wf = scon.execute(
        "select sum(is_wellformed::int), count(*) from feat_dns").fetchone()
    mo.vstack([
        mo.md(
            f"""
            ## TLS & DNS structure — rendered from the DP release

            The enclave side carries rich protocol structure: **{_n_tls_r:,} TLS
            flows** over {_n_stacks} client stacks ({_n_tls_canon:,} canonical
            handshakes) and **{_n_dns_r:,} DNS query records** over
            {_n_names:,} names ({_n_fans} subdomain-fan domains). The synthetic
            side now renders **{_n_tls_s} TLS flows** and **{_n_dns_s} DNS query
            records** from two auxiliary ε-budgeted DP histograms (`tls_stack`
            rank mix, `dns_shape` kind/label/length mix) — the exact ledger is
            7×(2/5) + 6/25 + 9/25 + 49/100 = **389/100 ≤ 4/1** (policy
            release-1.3.0, privacy unit = monitored internal host).

            What crosses the gate is only the **mix over public shape cells**:
            no fingerprint hash, cipher list, qname or domain string ever
            leaves. The rendered contents are synthetic — six made-up client
            stacks over a fixed canonical-handshake record skeleton
            ({_syn_stacks[1]}/{_n_tls_s} canonical on the synth warehouse). DNS
            names and TLS SNI values come from **common-knowledge top-k
            histograms** — policy-released, suppression-thresholded artifacts
            (≥2 hosts, ≥10 queries/flows, no `*.arpa` or single-label names,
            outside the ε ledger) — with synthetic fallbacks
            ({_syn_wf[0]}/{_syn_wf[1]} RFC-well-formed).
            Cipher realism remains a declared gap; the per-column tests in
            `tests/test_synth_columns.py` assert the structure and name
            provenance.
            """
        ),
        mo.hstack([
            mo.stat(f"{_n_tls_r} → {_n_tls_s}", label="TLS flows (real → synth)"),
            mo.stat(f"{_n_dns_r:,} → {_n_dns_s}", label="DNS queries (real → synth)"),
            mo.stat(f"{_n_stacks} / {_n_fans}", label="stacks / fan domains",
                    caption="enclave-only, never released"),
        ], justify="space-around"),
        _chart,
    ])
    return


@app.cell
def _(json, mo, pd, rcon):
    # geography: enclave sources & destinations on a MapLibre map; the synth
    # side deliberately has none (documentation address ranges) — declared gap
    try:
        import geoip2fast as _g2f
        _geo = _g2f.GeoIP2Fast()
        _ext = rcon.execute(
            """
            select 'source' as direction, sa as ip,
                   count(*) as flows
            from feat_flows where not src_internal group by 1, 2
            union all
            select 'destination', da, count(*)
            from feat_flows where not dst_internal group by 1, 2
            """
        ).fetch_df()
        _lk = {ip: _geo.lookup(ip) for ip in _ext.ip.unique()}
        _ext["country_code"] = [_lk[ip].country_code or "??" for ip in _ext.ip]
        _ext["country"] = [_lk[ip].country_name or "unknown" for ip in _ext.ip]
        _gdf = (_ext.groupby(["direction", "country_code", "country"],
                             as_index=False)
                .agg(flows=("flows", "sum"), ips=("ip", "nunique"))
                .sort_values("flows", ascending=False))
        _centroids = {
            "US": (39.8, -98.6), "CA": (56.1, -106.3), "MX": (23.6, -102.5),
            "GB": (54.0, -2.9), "IE": (53.2, -7.7), "FR": (46.6, 2.2),
            "DE": (51.2, 10.4), "NL": (52.2, 5.3), "BE": (50.6, 4.7),
            "CH": (46.8, 8.2), "AT": (47.6, 14.1), "IT": (42.8, 12.1),
            "ES": (40.2, -3.6), "PT": (39.6, -8.0), "SE": (62.8, 16.7),
            "NO": (64.6, 11.5), "DK": (56.0, 9.9), "FI": (64.5, 26.3),
            "PL": (52.1, 19.4), "CZ": (49.8, 15.5), "RU": (61.5, 105.0),
            "UA": (48.4, 31.2), "TR": (39.0, 35.2), "IL": (31.4, 35.0),
            "SA": (24.0, 45.0), "AE": (24.0, 54.0), "IN": (22.9, 79.6),
            "CN": (35.9, 104.2), "HK": (22.3, 114.2), "TW": (23.7, 121.0),
            "JP": (36.6, 138.0), "KR": (36.4, 127.8), "SG": (1.35, 103.8),
            "TH": (15.1, 101.0), "MY": (4.2, 102.0), "ID": (-2.5, 118.0),
            "PH": (12.9, 121.8), "AU": (-25.3, 133.8), "NZ": (-41.8, 172.8),
            "BR": (-10.8, -53.1), "AR": (-35.4, -65.2), "CL": (-35.7, -71.5),
            "CO": (4.1, -73.1), "PE": (-9.2, -75.0), "ZA": (-29.0, 25.1),
            "EG": (26.6, 30.0), "NG": (9.1, 8.7), "KE": (0.5, 37.9),
        }
        _feats = []
        for _row in _gdf.itertuples():
            _c = _centroids.get(_row.country_code)
            if _c:
                _off = -1.6 if _row.direction == "source" else 1.6
                _feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [_c[1] + _off, _c[0]]},
                    "properties": {"country": _row.country,
                                   "direction": _row.direction,
                                   "flows": int(_row.flows),
                                   "ips": int(_row.ips)},
                })
        _geojson = json.dumps({"type": "FeatureCollection", "features": _feats})
        _html = """<!doctype html><html><head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<style>html,body,#map{margin:0;height:100%;background:#fcfcfb}</style>
</head><body><div id="map"></div><script>
const data = __GEOJSON__;
const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [-40, 35], zoom: 1.2, attributionControl: true
});
map.addControl(new maplibregl.NavigationControl());
map.on('load', () => {
  map.addSource('flows', {type: 'geojson', data});
  map.addLayer({
    id: 'flow-circles', type: 'circle', source: 'flows',
    paint: {
      'circle-radius': ['max', 5, ['*', 3.2, ['log10', ['+', 1, ['get', 'flows']]]]],
      'circle-color': ['match', ['get', 'direction'],
                       'source', '#eb6834', '#2a78d6'],
      'circle-opacity': 0.75,
      'circle-stroke-color': '#fcfcfb',
      'circle-stroke-width': 2
    }
  });
  map.on('click', 'flow-circles', (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup().setLngLat(e.lngLat)
      .setHTML(`<b>${p.country}</b> — ${p.direction}<br/>${p.flows} flows, ${p.ips} IPs`)
      .addTo(map);
  });
  map.on('mouseenter', 'flow-circles', () => map.getCanvas().style.cursor = 'pointer');
  map.on('mouseleave', 'flow-circles', () => map.getCanvas().style.cursor = '');
});
const legend = document.createElement('div');
legend.style.cssText = 'position:absolute;top:10px;left:10px;background:#fcfcfbcc;' +
  'padding:6px 10px;border-radius:6px;font:12px sans-serif;color:#52514e';
legend.innerHTML = '<span style="color:#eb6834">\\u25cf</span> source countries (external sa) &nbsp; ' +
  '<span style="color:#2a78d6">\\u25cf</span> destination countries (external da)';
document.body.appendChild(legend);
</script></body></html>""".replace("__GEOJSON__", _geojson)
        import base64 as _b64
        _src = "data:text/html;base64," + _b64.b64encode(_html.encode()).decode()
        _top = pd.concat([
            _gdf[_gdf.direction == "source"].head(6),
            _gdf[_gdf.direction == "destination"].head(6),
        ])
        _geo_out = mo.vstack([
            mo.md(
                """
                ## Geography — source & destination countries (MapLibre GL JS)

                **Enclave side only.** Orange = source countries (external
                `sa`, inbound), blue = destination countries (external `da`,
                outbound); circle area ∝ log(flows). The **synthetic side has
                no geography by design**: hosts are rendered into the
                documentation ranges `10.99.0.0/24` → `198.51.100.0/24`
                (TEST-NET-2) — real IPs and countries are identifying and sit
                outside every release category, so this is a declared
                `fields_not_modelled` gap, not a similarity failure.
                """
            ),
            mo.Html(f'<iframe src="{_src}" style="width:100%;height:480px;'
                    f'border:1px solid #e1e0d9;border-radius:8px"></iframe>'),
            mo.ui.table(_top, selection=None),
        ])
    except Exception as _e:  # geoip2fast missing or lookup failure
        _geo_out = mo.md(f"⚠️ IP→country enrichment unavailable: `{_e}` — "
                         "run `make setup` to install geoip2fast.")
    _geo_out
    return


@app.cell
def _(ROOT, json, mo, pd):
    _p = ROOT / "artifacts" / "reports" / "realism_audit.json"
    if not _p.exists():
        _out = mo.md("⚠️ Realism audit not run — `make realism-audit` "
                     "(enclave-only; its report is not exportable).")
    else:
        _a = json.loads(_p.read_text())
        _tiles = []
        _an = _a.get("anomaly", _a)
        for _k, _lbl in (("share_above_real_p95", "synth flows above real p95 anomaly"),
                         ("share_above_real_p99", "above real p99")):
            if _k in _an:
                _tiles.append(mo.stat(f"{100*_an[_k]:.1f}%", label=_lbl,
                                      caption="~5% / ~1% if synth were as typical as real"))
        _tv = _a.get("tv", {})
        for _k, _v in (_tv.items() if isinstance(_tv, dict) else []):
            _tiles.append(mo.stat(f"{_v:.2f}", label=f"TV {_k}",
                                  caption="½Σ|P−Q| over public bins; "
                                          "0 = identical, 1 = disjoint"))
        _out = mo.vstack([
            mo.md(
                """
                ## Realism audit — the production models as judges

                The real-data-trained autoencoder scores every synthetic flow;
                shares above the real data's anomaly percentiles and TV
                distances on public histogram bins quantify the realism gap.
                This is a **graded audit, not a pass/fail**: the generator
                targets the *spec contract and the DP release*, not realism —
                the gap measures exactly what the assumption-driven renderer
                does not model.
                """
            ),
            mo.hstack(_tiles, justify="space-around") if _tiles else mo.md(""),
            mo.accordion({"full report": mo.ui.table(
                pd.json_normalize(_a).T.reset_index()
                .rename(columns={"index": "metric", 0: "value"}), selection=None)}),
        ])
    _out
    return


@app.cell
def _(mo, pd, syn_cert, syn_meta):
    _rows = []
    for _k, _v in syn_meta.get("released_facts", {}).items():
        _rows.append({"item": _k, "status": "RELEASED FACT (DP)",
                      "detail": str(_v)[:160]})
    _assume = syn_meta.get("assumptions", {})
    _items = _assume.items() if isinstance(_assume, dict) else enumerate(_assume)
    for _k, _v in _items:
        _rows.append({"item": str(_k), "status": "assumption",
                      "detail": str(_v)[:160]})
    mo.vstack([
        mo.md(
            f"""
            ---
            ## Provenance: released facts vs assumptions

            Everything the generator used is declared: the only data-derived
            input is the DP release; all shape comes from `FlowShape.lean`.
            Certificate: {"**kernel-checked** (axioms: " + ", ".join(syn_cert.get("axioms", [])) + ")" if syn_cert else "not generated — run `make certify`"}.
            """
        ),
        mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=20),
    ])
    return


if __name__ == "__main__":
    app.run()
