# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb"]
# ///
"""Synth-vs-real comparison notebook: point/trajectory volumes, mode and
motif mixes, step-speed and turn-angle histograms recomputed from the
synthetic TSV, GPS-noise wobble vs the released moments, the Tokyo/Beijing
geography assertion panel, a per-column TV scorecard with each column's fill
source, and the declared generation gaps. Declared-gap framing throughout."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import pandas as pd

    alt.data_transformers.enable("default", max_rows=20_000)
    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT / "python"))
    from spec import (  # noqa: E402
        DEP_BINS, DWELL_CLASSES, DIST_CLASSES, MODE_CLASSES, MOTIF_CLASSES,
        SPD_BINS, SPEED_CLASSES, TURN_BINS, load_spec, mode_map,
    )
    from maplibre_widget import (  # noqa: E402
        BASEMAPS, MapLibreMap, default_basemap_label, feature_collection,
    )

    SPEC = load_spec()
    RAW2CLS = mode_map()
    return (BASEMAPS, Counter, DEP_BINS, DIST_CLASSES, DWELL_CLASSES,
            MODE_CLASSES, MOTIF_CLASSES, MapLibreMap, RAW2CLS, ROOT,
            SPD_BINS, SPEC, SPEED_CLASSES, TURN_BINS, alt,
            default_basemap_label, duckdb, feature_collection, json, mo, pd)


@app.cell
def _(ROOT, SPEC, duckdb, json):
    def _load_json(p):
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    # the real warehouse (read-only; may be absent on a fresh checkout)
    real_con = None
    try:
        real_con = duckdb.connect(str(ROOT / "artifacts" / "enclave.duckdb"),
                                  read_only=True)
    except duckdb.Error:
        pass

    # the synthetic TSV -> an in-memory step table (dt, displacement, speed)
    _tsv = ROOT / "synth" / "geolife_synth.tsv"
    synth_con = None
    if _tsv.exists():
        synth_con = duckdb.connect()
        synth_con.execute(f"""
            create table synth_steps as
            with p as (
                select user_id, sequence_uuid, trajectory_id, mode,
                       -- TSV is UTC (source convention); shift to the
                       -- Beijing-local frame staging uses for hour bins
                       strptime(datetime, '%Y-%m-%d %H:%M:%S')
                           + to_seconds({int(SPEC['source_utc_offset_s'])}) dtt,
                       cast(latitude as double) lat,
                       cast(longitude as double) lon
                from read_csv('{_tsv.as_posix()}', delim='\t', header=true,
                              all_varchar=true)
            ), q as (
                select *, epoch(dtt)::bigint ts from p
            ), r as (
                select user_id, sequence_uuid, trajectory_id, mode, dtt, ts,
                       lat, lon,
                       ts - lag(ts) over w as dt,
                       (lon - lag(lon) over w) * 111320
                           * cos(radians(lat)) as dx,
                       (lat - lag(lat) over w) * 110540 as dy
                from q
                window w as (partition by user_id, trajectory_id order by ts)
            )
            select *, sqrt(dx*dx + dy*dy) as dist,
                   case when dt > 0 then sqrt(dx*dx + dy*dy) / dt end as speed
            from r""")

    # scratch warehouse written by `scripts/synth.py validate` (the synthetic
    # TSV re-ingested through the PRODUCTION dbt pipeline)
    scratch_con = None
    _scratch = ROOT / "synth" / "scratch.duckdb"
    if _scratch.exists():
        try:
            scratch_con = duckdb.connect(str(_scratch), read_only=True)
        except duckdb.Error:
            scratch_con = None

    plan = _load_json(ROOT / "synth" / "plan.json")
    meta = _load_json(ROOT / "synth" / "meta.json")
    genspec = _load_json(ROOT / "lean" / "out" / "genspec.json")
    return genspec, meta, plan, real_con, scratch_con, synth_con


@app.cell
def _(mo, synth_con):
    mo.md(
        "# Synthetic vs real trajectories — scorecard\n"
        "The generator targets the **spec contract and the DP release**, not "
        "realism; gaps are declared, measured and reported, never hidden. "
        "Every synthetic column is compared against the enclave warehouse "
        "and tagged with its fill source: DP release, category-2 suppressed "
        "aggregate, public prior, or declared gap."
        + ("" if synth_con is not None
           else "\n\n*(no synthetic data yet — run `make synth`)*"))
    return


@app.cell
def _(mo, pd, plan, real_con, synth_con):
    _out = mo.md("*(corpus stats appear after `make synth`)*")
    if synth_con is not None:
        def _row(metric, real, synth):
            return {"metric": metric, "real (enclave)": real,
                    "synthetic": synth}

        _r = [None] * 6
        if real_con is not None:
            _r = list(real_con.execute(
                "select count(distinct user_id), count(*), "
                "count(distinct trajectory_id), "
                "strftime(min(datetime), '%Y-%m-%d'), "
                "strftime(max(datetime), '%Y-%m-%d'), null "
                "from stg_points").fetchone())
            _r[5] = real_con.execute(
                "select median(dt_s) from feat_steps "
                "where dt_s is not null").fetchone()[0]
        _s = list(synth_con.execute(
            "select count(distinct user_id), count(*), "
            "count(distinct trajectory_id), "
            "strftime(min(dtt), '%Y-%m-%d'), strftime(max(dtt), '%Y-%m-%d'), "
            "median(dt) filter (dt is not null) from synth_steps").fetchone())
        _rows = [
            _row("users", _r[0], _s[0]),
            _row("GPS points", _r[1], _s[1]),
            _row("trajectories", _r[2], _s[2]),
            _row("points per user (mean)",
                 None if not _r[0] else round(_r[1] / _r[0]),
                 round(_s[1] / max(1, _s[0]))),
            _row("first day", _r[3], _s[3]),
            _row("last day", _r[4], _s[4]),
            _row("median sampling dt (s)", _r[5], _s[5]),
            _row("planned days per user", "—",
                 (plan or {}).get("n_days", "?")),
        ]
        _out = mo.vstack([
            mo.md("## Volumes and cadence\n"
                  "The synthetic corpus is deliberately smaller and tamer: "
                  "a fixed two-week slice per user (public prior), device "
                  "off during long stays and home days (declared), "
                  "render-grid cadence rather than the real 1–2 s logger."),
            mo.ui.table(pd.DataFrame(_rows), selection=None),
        ])
    _out
    return


@app.cell
def _(Counter, MODE_CLASSES, RAW2CLS, alt, mo, pd, real_con, synth_con):
    _out = mo.md("*(mode mix appears after `make synth`)*")
    if synth_con is not None and real_con is not None:
        _real = Counter(dict(real_con.execute(
            "select mode_class, count(*) from stg_points group by 1"
        ).fetchall()))
        _syn = Counter()
        for _raw, _n in synth_con.execute(
                "select mode, count(*) from synth_steps group by 1").fetchall():
            _syn[RAW2CLS.get(_raw, "XX")] += _n
        _rows = []
        for _src, _c in (("real", _real), ("synthetic", _syn)):
            _tot = max(1, sum(_c.values()))
            for _b in MODE_CLASSES:
                _rows.append({"src": _src, "mode_class": _b,
                              "share": _c.get(_b, 0) / _tot})
        _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
            x=alt.X("mode_class:N", sort=MODE_CLASSES),
            y=alt.Y("share:Q", title="share of points"),
            color="src:N", xOffset="src:N").properties(
            width=480, height=220,
            title="Mode mix: share of GPS points per public mode class")
        _out = mo.vstack([
            mo.md("## Mode mix (from TSV labels via the seed map)\n"
                  "Synthetic raw mode labels are mapped through the same "
                  "`dbt/seeds/mode_map.csv` the pipeline uses. The synthetic "
                  "mix is drawn from the **DP mode histogram** per trip; "
                  "stay wobble is emitted as `walk`, which inflates the "
                  "synthetic walk share (declared rendering convention)."),
            mo.ui.altair_chart(_chart),
        ])
    _out
    return


@app.cell
def _(Counter, MOTIF_CLASSES, alt, mo, pd, plan, real_con, scratch_con,
      synth_con):
    _out = mo.md("*(motif mix appears after `make synth`)*")
    if synth_con is not None and real_con is not None:
        _real = Counter(dict(real_con.execute(
            "select motif_class, count(*) from feat_user_days group by 1"
        ).fetchall()))
        _syn, _basis = Counter(), None
        if scratch_con is not None:
            _syn = Counter(dict(scratch_con.execute(
                "select motif_class, count(*) from feat_user_days group by 1"
            ).fetchall()))
            _basis = ("re-detected: the synthetic TSV re-ingested through "
                      "the PRODUCTION dbt pipeline (`synth/scratch.duckdb`)")
        elif plan is not None:
            for _u in plan.get("users", []):
                _syn.update(_u.get("motifs", []))
            _basis = ("planted: the generator's latent plan "
                      "(`synth/plan.json`) — run `make synth` validation "
                      "for the re-detected mix")
        if _basis is not None:
            _rows = []
            for _src, _c in (("real", _real), ("synthetic", _syn)):
                _tot = max(1, sum(_c.values()))
                for _b in MOTIF_CLASSES:
                    _rows.append({"src": _src, "motif": _b,
                                  "share": _c.get(_b, 0) / _tot})
            _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
                x=alt.X("motif:N", sort=MOTIF_CLASSES),
                y=alt.Y("share:Q", title="share of user-days"),
                color="src:N", xOffset="src:N").properties(
                width=480, height=220,
                title="Daily-motif mix: real vs synthetic")
            _out = mo.vstack([
                mo.md("## Daily motifs\n"
                      f"Synthetic side **{_basis}**. The mix is drawn from "
                      "the **DP daily-motif histogram**; planted users 001 "
                      "(commuter) and 002 (revisit beacon + tour) are the "
                      "round-trip ground truth the validator must recall."),
                mo.ui.altair_chart(_chart),
            ])
    _out
    return


@app.cell
def _(SPEC, SPEED_CLASSES, TURN_BINS, alt, mo, pd, real_con, synth_con):
    _out = mo.md("*(kinematic histograms appear after `make synth`)*")
    if synth_con is not None and real_con is not None:
        _spd_case = (
            f"case when speed*1000 < {SPEC['spd_still_max_mmps']} "
            f"then 'spd_still' when speed*1000 < {SPEC['spd_walk_max_mmps']} "
            f"then 'spd_walk' when speed*1000 < {SPEC['spd_city_max_mmps']} "
            f"then 'spd_city' when speed*1000 < {SPEC['spd_fast_max_mmps']} "
            f"then 'spd_fast' else 'spd_vfast' end")
        _rows = []
        for _src, _pairs in (
            ("real", real_con.execute(
                "select speed_class, count(*) from feat_steps "
                "where speed_class is not null group by 1").fetchall()),
            ("synthetic", synth_con.execute(
                f"select {_spd_case}, count(*) from synth_steps "
                f"where speed is not null group by 1").fetchall()),
        ):
            _tot = max(1, sum(n for _, n in _pairs))
            _d = dict(_pairs)
            for _b in SPEED_CLASSES:
                _rows.append({"src": _src, "bin": _b,
                              "share": _d.get(_b, 0) / _tot})
        _speed = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
            x=alt.X("bin:N", sort=SPEED_CLASSES, title="step-speed class"),
            y=alt.Y("share:Q", title="share of steps"),
            color="src:N", xOffset="src:N").properties(
            width=320, height=220, title="Step speeds: real vs synth")
        _trows = []
        for _src, _pairs in (
            ("real", real_con.execute(
                "select turn_bin, n_points from mart_turn_hist").fetchall()),
            ("synthetic", synth_con.execute("""
                with mv as (
                    select user_id, trajectory_id, ts,
                           degrees(atan2(dx, dy)) heading
                    from synth_steps where dist >= 1
                ), t as (
                    select ((heading - lag(heading) over (
                        partition by user_id, trajectory_id order by ts)
                        + 540) % 360) - 180 as turn
                    from mv
                )
                select 'tb_' || least(floor((turn + 180) / 45), 7)::int,
                       count(*) from t where turn is not null group by 1
            """).fetchall()),
        ):
            _tot = max(1, sum(n for _, n in _pairs))
            _d = dict(_pairs)
            for _b in TURN_BINS:
                _trows.append({"src": _src, "bin": _b,
                               "share": _d.get(_b, 0) / _tot})
        _turn = alt.Chart(pd.DataFrame(_trows)).mark_bar().encode(
            x=alt.X("bin:N", sort=TURN_BINS, title="signed turn (45° bins)"),
            y=alt.Y("share:Q", title="share of moving steps"),
            color="src:N", xOffset="src:N").properties(
            width=320, height=220, title="Turn angles: real vs synth")
        _out = mo.vstack([
            mo.md("## Kinematic texture (recomputed from the TSV)\n"
                  "Both histograms are computed on the fly from the "
                  "synthetic points with the same public thresholds the "
                  "pipeline uses. Speeds come from **category-2 per-mode "
                  "kinematics** (speed caps are public physics); turn "
                  "innovations are drawn from the **released 45°-bin turn "
                  "histogram** with a forced-turn rule, so there are no "
                  "straight runs and no 0/±90° lattice turns."),
            mo.ui.altair_chart(_speed | _turn),
        ])
    _out
    return


@app.cell
def _(MODE_CLASSES, RAW2CLS, SPEC, alt, duckdb, mo, pd, real_con,
      scratch_con, synth_con):
    _out = mo.md("*(route-shape comparison appears after `make synth`)*")
    if synth_con is not None and real_con is not None:
        _blocks = [mo.md(
            "## Route shape: real vs synth\n"
            "**Straightness** (net displacement / path length per moving "
            "sequence, ≥"
            f"{int(SPEC['route_min_net_m'])} m net) is recomputed from the "
            "synthetic TSV on the fly — the same statistic the road-likeness "
            "test and `synth.py validate` enforce (a straight-vector "
            "renderer medians ≈ 1.0). The real side is the released "
            "`route_shape` aggregate the generator targets.")]
        # real per-mode straightness: the released category-2 aggregate
        _real_rs = None
        try:
            _real_rs = real_con.execute(
                "select mode_class, median_straightness "
                "from mart_route_shape").fetchall()
        except duckdb.Error:
            _blocks.append(mo.md("*(real `mart_route_shape` missing — "
                                 "rebuild the warehouse with `make train`)*"))
        # synth per-sequence straightness recomputed from the TSV
        _legs = synth_con.execute(f"""
            with legs as (
                select user_id, sequence_uuid, mode("mode") raw_mode,
                       sum(coalesce(dist, 0)) path_m,
                       arg_min(lat, ts) lat0, arg_min(lon, ts) lon0,
                       arg_max(lat, ts) lat1, arg_max(lon, ts) lon1
                from synth_steps
                group by user_id, sequence_uuid
            )
            select raw_mode,
                   least(1.0, sqrt(pow((lon1 - lon0) * 111320
                                       * cos(radians((lat0 + lat1) / 2)), 2)
                                   + pow((lat1 - lat0) * 110540, 2))
                         / path_m) straightness
            from legs
            where path_m > 0
              and sqrt(pow((lon1 - lon0) * 111320
                           * cos(radians((lat0 + lat1) / 2)), 2)
                       + pow((lat1 - lat0) * 110540, 2))
                  >= {int(SPEC['route_min_net_m'])}""").df()
        _legs["mode_class"] = _legs["raw_mode"].map(
            lambda m: RAW2CLS.get(m, "XX"))
        _srows = [{"src": "synthetic (TSV, recomputed)", "mode_class": _m,
                   "median_straightness": round(float(_g), 4),
                   "n_sequences": int(_n)}
                  for _m, _g, _n in _legs.groupby("mode_class")["straightness"]
                  .agg(["median", "count"]).reset_index().itertuples(
                      index=False) if _m in MODE_CLASSES]
        if _real_rs:
            _srows += [{"src": "real (released route_shape)",
                        "mode_class": _m,
                        "median_straightness": round(float(_v), 4),
                        "n_sequences": None} for _m, _v in _real_rs]
        if _srows:
            _sdf = pd.DataFrame(_srows)
            _schart = alt.Chart(_sdf).mark_bar().encode(
                x=alt.X("mode_class:N", sort=MODE_CLASSES),
                y=alt.Y("median_straightness:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title="median straightness (net / path)"),
                color="src:N", xOffset="src:N",
                tooltip=["src", "mode_class", "median_straightness",
                         "n_sequences"]).properties(
                width=480, height=220,
                title="Per-mode median straightness: real vs synth")
            _blocks.append(mo.ui.altair_chart(_schart))
        # seg/junction histograms: released vs re-detected (the production
        # junction detector over the synthetic TSV, from scratch.duckdb)
        _scr_ok = False
        if scratch_con is not None:
            try:
                scratch_con.execute(
                    "select 1 from feat_route_segments limit 1")
                _scr_ok = True
            except duckdb.Error:
                pass
        if _scr_ok:
            _hrows = []
            for _grid, _col, _real_sql, _scr_sql in (
                (["seg_lt50", "seg_50_150", "seg_150_600", "seg_ge600"],
                 "seg_class",
                 "select seg_class, n_segments from mart_seg_hist",
                 "select seg_class, count(*) from feat_route_segments "
                 "group by 1"),
                (["ja_30_60", "ja_60_120", "ja_120_180"], "ja_class",
                 "select ja_class, n_junctions from mart_junction_hist",
                 "select ja_class, count(*) from feat_route_segments "
                 "group by 1"),
            ):
                for _src, _con_, _sql in (
                        ("real (released)", real_con, _real_sql),
                        ("synthetic (re-detected)", scratch_con, _scr_sql)):
                    try:
                        _d = dict(_con_.execute(_sql).fetchall())
                    except duckdb.Error:
                        continue
                    _tot = max(1, sum(_d.values()))
                    for _b in _grid:
                        _hrows.append({"src": _src, "hist": _col, "bin": _b,
                                       "share": _d.get(_b, 0) / _tot})
            if _hrows:
                _hdf = pd.DataFrame(_hrows)
                _hchart = alt.Chart(_hdf).mark_bar().encode(
                    x=alt.X("bin:N", sort=None, title=None),
                    y=alt.Y("share:Q", title="share"),
                    color="src:N", xOffset="src:N").properties(
                    width=300, height=200).facet(
                    column=alt.Column("hist:N", title=None)).resolve_scale(
                    x="independent")
                _blocks.append(mo.md(
                    "**Segment-length and junction-angle mixes** — the "
                    "released histograms the generator samples from vs the "
                    "production junction detector re-run over the synthetic "
                    "TSV (`synth/scratch.duckdb`). Drift here is soft by "
                    "design: resampling the rendered polyline re-quantizes "
                    "segment lengths."))
                _blocks.append(mo.ui.altair_chart(_hchart))
        else:
            _blocks.append(mo.md(
                "*(seg/junction re-detection comparison appears after "
                "`make synth` validation rebuilds `synth/scratch.duckdb` "
                "with the route models)*"))
        _out = mo.vstack(_blocks)
    _out
    return


@app.cell
def _(genspec, mo, pd, scratch_con, synth_con):
    _out = mo.md("*(noise comparison appears after `make synth`)*")
    if synth_con is not None:
        _rows = []
        if genspec is not None and "noise" in genspec:
            _sd = genspec["noise"].get("sd_mm")
            _l1 = genspec["noise"].get("lag1_pct")
            if _sd is not None:
                _rows.append({"quantity": "released per-axis wobble sd (m)",
                              "source": "genspec.json (category-2 tight "
                                        "noise moments)",
                              "value": round(_sd / 1000, 1)})
            if _l1 is not None:
                _rows.append({"quantity": "released lag-1 autocorrelation",
                              "source": "genspec.json",
                              "value": round(_l1 / 100, 2)})
        else:
            _rows.append({"quantity": "released wobble moments",
                          "source": "lean/out/genspec.json",
                          "value": "missing — run `make distill`"})
        if scratch_con is not None:
            for _m, _v in scratch_con.execute(
                    "select metric, value from mart_noise_moments "
                    "where metric like '%resid_sd%' or metric like "
                    "'%resid_lag1_median%' order by metric").fetchall():
                _rows.append({"quantity": f"synth re-detected {_m}",
                              "source": "scratch.duckdb (production "
                                        "pipeline over the synth TSV)",
                              "value": round(_v, 2)})
        else:
            _approx = synth_con.execute("""
                with xy as (
                    select user_id, trajectory_id,
                           round(lat * 500) / 500 cla,
                           round(lon * 500) / 500 clo,
                           lon * 111320 * cos(radians(lat)) x,
                           lat * 110540 y
                    from synth_steps where speed is not null and speed < 2
                ), cl as (
                    select sqrt(var_pop(x) + var_pop(y)) rms
                    from xy group by user_id, trajectory_id, cla, clo
                    having count(*) >= 10
                )
                select median(rms), count(*) from cl""").fetchone()
            if _approx and _approx[0] is not None:
                _rows.append({
                    "quantity": "synth stationary-cluster radial rms (m), "
                                "approx",
                    "source": f"TSV on the fly ({_approx[1]} clusters of "
                              "near-still points; run validate for the "
                              "pipeline's own estimate)",
                    "value": round(_approx[0], 2)})
        _out = mo.vstack([
            mo.md("## GPS-noise wobble vs the released moments\n"
                  "The generator's stay wobble is an AR(1) per axis matched "
                  "to the released tight-subset noise moments, clamped into "
                  "the staypoint detection box (`wobble_box_fits`). The "
                  "released per-axis sd implies a radial rms of sd·√2; the "
                  "re-detected `resid_sd_m` from the scratch warehouse is "
                  "the apples-to-apples readback."),
            mo.ui.table(pd.DataFrame(_rows), selection=None),
        ])
    _out
    return


@app.cell
def _(SPEC, mo, pd, synth_con):
    _out = mo.md("*(geography assertions appear after `make synth`)*")
    if synth_con is not None:
        _mn_la, _mx_la, _mn_lo, _mx_lo, _n = synth_con.execute(
            "select min(lat), max(lat), min(lon), max(lon), count(*) "
            "from synth_steps").fetchone()
        _ex = {k: SPEC[f"excl_{k}_cd"] / 100 for k in
               ("lat_min", "lat_max", "lon_min", "lon_max")}
        _tg = {k: SPEC[f"tgt_{k}_cd"] / 100 for k in
               ("lat_min", "lat_max", "lon_min", "lon_max")}
        _in_excl = synth_con.execute(
            "select count(*) from synth_steps where lat between ? and ? "
            "and lon between ? and ?",
            [_ex["lat_min"], _ex["lat_max"],
             _ex["lon_min"], _ex["lon_max"]]).fetchone()[0]
        _out_tgt = synth_con.execute(
            "select count(*) from synth_steps where not (lat between ? and ? "
            "and lon between ? and ?)",
            [_tg["lat_min"], _tg["lat_max"],
             _tg["lon_min"], _tg["lon_max"]]).fetchone()[0]
        _df = pd.DataFrame([
            {"box": "synthetic data extent", "lat_min": round(_mn_la, 4),
             "lat_max": round(_mx_la, 4), "lon_min": round(_mn_lo, 4),
             "lon_max": round(_mx_lo, 4)},
            {"box": "Tokyo target (every point must be inside)", **_tg},
            {"box": "Beijing exclusion (zero points allowed)", **_ex},
        ])
        _b1 = (mo.callout(mo.md(f"**PASS** — 0 of {_n:,} synthetic points "
                                f"fall in the Beijing exclusion box."),
                          kind="success") if _in_excl == 0 else
               mo.callout(mo.md(f"**FAIL** — {_in_excl:,} synthetic points "
                                f"fall in the Beijing exclusion box."),
                          kind="danger"))
        _b2 = (mo.callout(mo.md(f"**PASS** — all {_n:,} points are inside "
                                f"the Tokyo metro target box."),
                          kind="success") if _out_tgt == 0 else
               mo.callout(mo.md(
                   f"**FAIL** — {_out_tgt:,} points lie outside the Tokyo "
                   f"target box. The raw generator emits a neutral local "
                   f"frame (around 0°N 0°E); this panel passes only after "
                   f"the mandatory `manipulate` juxtaposition step."),
                   kind="danger"))
        _out = mo.vstack([
            mo.md("## Geography assertions (the manipulate contract)\n"
                  "No source geography is ever input to the generator, so "
                  "none can leak: the synthetic output must contain **zero** "
                  "points in the Beijing exclusion box and land entirely in "
                  "the Tokyo metro target box (both boxes are public spec "
                  "constants; their disjointness is a kernel-checked "
                  "theorem, `excl_tgt_disjoint`)."),
            mo.ui.table(_df, selection=None),
            _b1, _b2,
        ])
    _out
    return


@app.cell
def _(feature_collection, synth_con):
    synth_tracks_fc = None
    synth_track_bounds = None
    synth_track_itv_s = 30
    n_synth_track_legs = 0
    n_synth_track_coords = 0
    if synth_con is not None:
        _cap = 150_000

        def _kept_sql(itv):
            return f"""
            with pts as (
                select user_id, sequence_uuid, trajectory_id, ts, lat, lon,
                       mode,
                       min(ts) over (partition by user_id,
                                     sequence_uuid) t0
                from synth_steps
            ), marked as (
                select *,
                       row_number() over (
                           partition by user_id, sequence_uuid,
                                        (ts - t0) // {itv}
                           order by ts) rn,
                       row_number() over (
                           partition by user_id, sequence_uuid
                           order by ts desc) rl
                from pts
            )
            select * from marked where rn = 1 or rl = 1"""

        _n = synth_con.execute(
            f"select count(*) from ({_kept_sql(synth_track_itv_s)})"
        ).fetchone()[0]
        while _n > _cap:
            synth_track_itv_s = synth_track_itv_s * _n // _cap + 1
            _n = synth_con.execute(
                f"select count(*) from ({_kept_sql(synth_track_itv_s)})"
            ).fetchone()[0]
        _legs = synth_con.execute(f"""
            select user_id, any_value(trajectory_id) trajectory,
                   mode("mode") raw_mode, count(*) n,
                   round((max(ts) - min(ts)) / 60.0, 1) mins,
                   list([round(lon, 5), round(lat, 5)] order by ts) coords
            from ({_kept_sql(synth_track_itv_s)})
            group by user_id, sequence_uuid
            having count(*) >= 2""").fetchall()
        synth_tracks_fc = feature_collection([
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": _c},
             "properties": {"mode": _m, "user": _u, "trajectory": _t,
                            "points": _np, "minutes": _mins}}
            for _u, _t, _m, _np, _mins, _c in _legs])
        n_synth_track_legs = len(_legs)
        n_synth_track_coords = sum(_r[3] for _r in _legs)
        synth_track_bounds = [round(_v, 4) for _v in synth_con.execute(
            "select min(lon), min(lat), max(lon), max(lat) "
            "from synth_steps").fetchone()]
    return (n_synth_track_coords, n_synth_track_legs, synth_track_bounds,
            synth_track_itv_s, synth_tracks_fc)


@app.cell
def _(BASEMAPS, default_basemap_label, mo):
    synth_basemap_pick = mo.ui.dropdown(
        options=list(BASEMAPS), value=default_basemap_label(),
        label="basemap")
    return (synth_basemap_pick,)


@app.cell
def _(BASEMAPS, MapLibreMap, mo, n_synth_track_coords, n_synth_track_legs,
      synth_basemap_pick, synth_track_bounds, synth_track_itv_s,
      synth_tracks_fc):
    _out = mo.md("*(the synthetic trajectory map appears after "
                 "`make synth`)*")
    if synth_tracks_fc is not None:
        _palette = [("walk", "#f58518"), ("bike", "#54a24b"),
                    ("bus", "#4c78a8"), ("car", "#e45756"),
                    ("train", "#9d755d"), ("airplane", "#72b7b2"),
                    ("other", "#bab0ac")]
        _legend = mo.Html("".join(
            f'<span style="display:inline-block;margin:0 12px 4px 0;'
            f'font:12px system-ui"><span style="display:inline-block;'
            f'width:12px;height:12px;background:{_c};border-radius:2px;'
            f'margin-right:4px;vertical-align:-1px"></span>{_m}</span>'
            for _m, _c in _palette))
        _map = mo.ui.anywidget(MapLibreMap(
            basemap=BASEMAPS[synth_basemap_pick.value],
            bounds=synth_track_bounds, height=620,
            layers={"tracks": synth_tracks_fc}, visible=["tracks"]))
        _out = mo.vstack([
            mo.md(
                f"## Synthetic trajectory map (safe to show individually)\n"
                f"These are **synthetic, Tokyo-placed trajectories** — no "
                f"real user's movement is drawn, so individual tracks are "
                f"fine to render; the Beijing-exclusion assertion panel "
                f"above is the formal guarantee that none of the source "
                f"geography leaked. One line per generated recording "
                f"sequence, colored by the raw modality label the "
                f"generator wrote: {n_synth_track_legs:,} sequences / "
                f"{n_synth_track_coords:,} points, each sequence "
                f"downsampled to one point per {synth_track_itv_s} s "
                f"bucket (endpoints kept). The basemap fetches tiles for "
                f"the viewed area; choose \"none (no network)\" for a "
                f"fully offline map."),
            synth_basemap_pick,
            _legend,
            _map,
        ])
    _out
    return


@app.cell
def _(Counter, DEP_BINS, DIST_CLASSES, DWELL_CLASSES, MODE_CLASSES,
      MOTIF_CLASSES, RAW2CLS, SPD_BINS, SPEC, SPEED_CLASSES, TURN_BINS, alt,
      mo, pd, plan, real_con, scratch_con, synth_con):
    _out = mo.md("*(run `make synth` for the TV scorecard)*")
    if synth_con is not None and real_con is not None:
        def _tv(a, b, grid):
            ta, tb = sum(a.values()), sum(b.values())
            if not ta or not tb:
                return None
            return 0.5 * sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb)
                             for k in grid)

        def _cnt(con, sql):
            return Counter(dict(con.execute(sql).fetchall()))

        _spd_case = (
            f"case when speed*1000 < {SPEC['spd_still_max_mmps']} "
            f"then 'spd_still' when speed*1000 < {SPEC['spd_walk_max_mmps']} "
            f"then 'spd_walk' when speed*1000 < {SPEC['spd_city_max_mmps']} "
            f"then 'spd_city' when speed*1000 < {SPEC['spd_fast_max_mmps']} "
            f"then 'spd_fast' else 'spd_vfast' end")
        _hour_case = (
            f"case when %s <= {SPEC['dep_night_max_h']} then 'dep_night' "
            f"when %s <= {SPEC['dep_morning_max_h']} then 'dep_morning' "
            f"when %s <= {SPEC['dep_afternoon_max_h']} then 'dep_afternoon' "
            f"else 'dep_evening' end")
        _h = _hour_case.replace("%s", "hour(dtt)")

        _syn_mode = Counter()
        for _raw, _n in synth_con.execute(
                "select mode, count(*) from synth_steps group by 1").fetchall():
            _syn_mode[RAW2CLS.get(_raw, "XX")] += _n
        _syn_motif = Counter()
        if scratch_con is not None:
            _syn_motif = _cnt(scratch_con, "select motif_class, count(*) "
                                           "from feat_user_days group by 1")
            _motif_basis = "re-detected (scratch.duckdb)"
        elif plan is not None:
            for _u in plan.get("users", []):
                _syn_motif.update(_u.get("motifs", []))
            _motif_basis = "planted plan (plan.json)"
        else:
            _motif_basis = "unavailable"

        _score = [
            ("mode_class", MODE_CLASSES,
             _cnt(real_con,
                  "select mode_class, count(*) from stg_points group by 1"),
             _syn_mode, "DP release: mode histogram (clip 10)",
             "TSV labels via seed map"),
            ("motif_class", MOTIF_CLASSES,
             _cnt(real_con, "select motif_class, count(*) "
                            "from feat_user_days group by 1"),
             _syn_motif, "DP release: daily-motif histogram",
             _motif_basis),
            ("speed_class", SPEED_CLASSES,
             _cnt(real_con, "select speed_class, count(*) from feat_steps "
                            "where speed_class is not null group by 1"),
             _cnt(synth_con, f"select {_spd_case}, count(*) from synth_steps "
                             f"where speed is not null group by 1"),
             "category-2: per-mode kinematics (suppressed)",
             "recomputed from TSV steps"),
            ("turn_bin", TURN_BINS,
             _cnt(real_con, "select turn_bin, n_points from mart_turn_hist"),
             _cnt(synth_con, """
                with mv as (select user_id, trajectory_id, ts,
                                   degrees(atan2(dx, dy)) heading
                            from synth_steps where dist >= 1),
                t as (select ((heading - lag(heading) over (
                          partition by user_id, trajectory_id order by ts)
                          + 540) % 360) - 180 turn from mv)
                select 'tb_' || least(floor((turn + 180) / 45), 7)::int,
                       count(*) from t where turn is not null group by 1"""),
             "category-2: 45°-bin turn histogram (suppressed)",
             "recomputed from TSV headings"),
            ("hour_bin (points)", DEP_BINS,
             _cnt(real_con, "select hour_bin, count(*) from feat_steps "
                            "where hour_bin is not null group by 1"),
             _cnt(synth_con, f"select {_h}, count(*) from synth_steps "
                             f"group by 1"),
             "DP departure bins + public uniform prior in-range",
             "recomputed from TSV timestamps"),
        ]
        if scratch_con is not None:
            for _col, _grid, _sql, _fill in (
                ("dwell_class", DWELL_CLASSES,
                 "select dwell_class, count(*) from feat_visits group by 1",
                 "DP release: dwell histogram"),
                ("arr_bin", DEP_BINS,
                 "select arr_bin, count(*) from feat_visits group by 1",
                 "DP release: departure histogram"),
                ("trans_cell", None,
                 "select trans_cell, count(*) from feat_visits "
                 "where trans_cell is not null group by 1",
                 "DP release: role-transition histogram"),
                ("dist_class", DIST_CLASSES,
                 "select dist_class, count(*) from feat_visits "
                 "where dist_class is not null group by 1",
                 "category-2: displacement mix (suppressed)"),
                ("spd_bin (sp/day)", SPD_BINS,
                 "select spd_bin, count(*) from feat_user_days group by 1",
                 "declared gap: implied by motif templates, not drawn"),
            ):
                _rc = _cnt(real_con, _sql)
                _sc = _cnt(scratch_con, _sql)
                _g = _grid if _grid is not None else sorted(set(_rc) | set(_sc))
                _score.append((_col, _g, _rc, _sc, _fill,
                               "re-detected (scratch.duckdb)"))
        _rows = []
        for _col, _grid, _rc, _sc, _fill, _basis in _score:
            _t = _tv(_rc, _sc, _grid)
            _rows.append({"column": _col,
                          "tv_real_vs_synth":
                              None if _t is None else round(_t, 4),
                          "fill_source": _fill, "synth_basis": _basis})
        _df = pd.DataFrame(_rows)
        _chart = alt.Chart(_df.dropna(subset=["tv_real_vs_synth"])).mark_bar(
        ).encode(
            x=alt.X("tv_real_vs_synth:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("column:N", sort="-x"),
            color=alt.Color("fill_source:N",
                            legend=alt.Legend(orient="bottom", columns=1)),
            tooltip=["column", "tv_real_vs_synth", "fill_source",
                     "synth_basis"]).properties(
            width=520, height=240, title="Per-column TV, real vs synth")
        _out = mo.vstack([
            mo.md(
                "## TV scorecard\n"
                "**TV = total variation distance** between the real and "
                "synthetic distributions over a column's public bins: "
                "TV = ½ · Σ|p_real(bin) − p_synth(bin)|, ranging **0 to 1** "
                "— 0 means identical mixes, 1 means completely disjoint. "
                "Read it as *the share of probability mass that would have "
                "to move to make the two distributions match* (TV 0.05 ⇒ 5% "
                "of mass misplaced). The same statistic gates `make synth` "
                "validation (soft tolerance 0.25 vs the genspec) — the "
                "generator targets the DP release and the motif templates, "
                "so some drift from the *real* data is expected and honest. "
                "Each row is tagged with the column's **fill source**: DP "
                "release, category-2 suppressed aggregate, public prior, or "
                "declared gap."),
            mo.ui.altair_chart(_chart),
            mo.ui.table(_df, selection=None),
        ])
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
    if not _items:
        _items = [mo.md("*(no reports yet — `make synth` writes "
                        "validation.json + meta.json; `make certify` the "
                        "kernel-checked certificate; `make realism-audit` "
                        "the model-based audit)*")]
    mo.vstack([mo.md("## Contracts, certificates and honest reports")] + _items)
    return


@app.cell
def _(meta, mo):
    _out = mo.md("")
    _gaps = ((meta or {}).get("assumptions", {}) or {}).get(
        "fields_not_modelled")
    if _gaps:
        _out = mo.md(
            "## Declared gaps (fields_not_modelled)\n" +
            "\n".join(f"- {g}" for g in _gaps) +
            "\n\nDeclared gaps over silent fabrication: each is measured by "
            "the realism audit, none is hidden behind plausible noise.")
    _out
    return


if __name__ == "__main__":
    app.run()
