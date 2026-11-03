# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "altair", "pandas", "duckdb"]
# ///
"""Enclave EDA notebook: GPS point volumes and quality, mode mix and per-mode
kinematics, staypoints and anchors, daily motifs, role/mode sequence
structure, GPS-noise moments and DP-release utility. Everything rendered here
is ENCLAVE-SIDE (per-user detail allowed); only the DP release section shows
what actually leaves. No individual trajectory is ever plotted — spatial
views are corpus-level density aggregates only."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import pandas as pd

    alt.data_transformers.enable("default", max_rows=20_000)
    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT / "python"))
    from maplibre_widget import (  # noqa: E402
        BASEMAPS, MapLibreMap, default_basemap_label, feature_collection,
    )

    con = duckdb.connect(str(ROOT / "artifacts" / "enclave.duckdb"),
                         read_only=True)
    return (BASEMAPS, MapLibreMap, ROOT, alt, con, default_basemap_label,
            feature_collection, json, mo, pd)


@app.cell
def _(mo):
    mo.md(
        """
        # Enclave mobility-trajectory encoder — EDA
        5,440,384 raw GPS points, 64 users, 4,345 labelled trajectories
        (2007-04-12 → 2011-12-31, GeoLife labelled subset, Beijing). All
        plots in this notebook are **enclave-only**; the final sections show
        the ε-DP release and the formal layer. Staypoint/anchor locations
        (`lat_c`/`lon_c`) never leave the enclave — the released geometry is
        anchor-relative by construction.
        """
    )
    return


@app.cell
def _(con, mo):
    _q = con.execute(
        "select counter, n, stage from q_data_integrity "
        "order by stage, counter").df()
    mo.vstack([
        mo.md("## Data integrity counters\n"
              "The staging funnel (5,440,384 raw rows → 4,865,675 kept: "
              "574,708 duplicate rows collapsed, 1 invalid coordinate "
              "dropped, 0 outside the public window) and the per-stage "
              "feature counts — the pipeline's own audit trail."),
        mo.ui.table(_q, selection=None),
    ])
    return


@app.cell
def _(alt, con, mo):
    _m = con.execute(
        "select mode_class, count(*) n from stg_points group by 1").df()
    _order = ["walk", "bike", "bus", "car", "rail", "other"]
    _chart = alt.Chart(_m).mark_bar().encode(
        x=alt.X("mode_class:N", sort=_order, title="mode class (public)"),
        y=alt.Y("n:Q", title="GPS points")).properties(
        width=480, height=240,
        title="Mode mix: points per public mode class (walk & bus dominate)")
    _kin = con.execute(
        "select mode_class, n_points, n_users, "
        "round(median_speed_mps,2) med_speed, round(p95_speed_mps,1) p95_speed, "
        "median_dt_s med_dt_s, round(median_abs_turn_deg,1) med_turn, "
        "round(median_straightness,2) med_straight, n_legs "
        "from mart_kinematics order by n_points desc").df()
    mo.vstack([
        mo.ui.altair_chart(_chart),
        mo.md("**Per-mode kinematics** (suppressed category-2 aggregate: "
              "≥5 users / ≥500 points per row). The physics reads correctly "
              "— walk 1.1 m/s, bike 3.1, bus 4.7, car 8.5, rail 19.1 median; "
              "turn magnitude falls as speed rises; `other` (airplane/boat) "
              "is fast, straight and sparsely sampled."),
        mo.ui.table(_kin, selection=None),
    ])
    return


@app.cell
def _(alt, con, mo):
    _s = con.execute(
        "select mode_class, speed_class, count(*) n from feat_steps "
        "where speed_class is not null group by 1, 2").df()
    _mode_order = ["walk", "bike", "bus", "car", "rail", "other"]
    _spd_order = ["spd_still", "spd_walk", "spd_city", "spd_fast", "spd_vfast"]
    _speed = alt.Chart(_s).transform_joinaggregate(
        tot="sum(n)", groupby=["mode_class"]).transform_calculate(
        share="datum.n / datum.tot").mark_bar().encode(
        x=alt.X("mode_class:N", sort=_mode_order, title="mode"),
        y=alt.Y("share:Q", stack="normalize", title="share of steps"),
        color=alt.Color("speed_class:N", sort=_spd_order),
        order=alt.Order("color_speed_class_sort_index:Q"),
        tooltip=["mode_class", "speed_class", "n"]).properties(
        width=300, height=220, title="Step-speed class by mode")
    _t = con.execute(
        "select turn_bin, n_points from mart_turn_hist order by turn_bin").df()
    _turn = alt.Chart(_t).mark_bar().encode(
        x=alt.X("turn_bin:N", title="signed turn, 45° bins (tb_3|tb_4 ≈ 0°)"),
        y=alt.Y("n_points:Q", title="steps")).properties(
        width=300, height=220, title="Turn-angle histogram (near-symmetric)")
    _c = con.execute(
        "select case when dt_s <= 1 then 'dt_1s' when dt_s <= 2 then 'dt_2s' "
        "when dt_s <= 5 then 'dt_3_5s' when dt_s <= 15 then 'dt_6_15s' "
        "when dt_s <= 60 then 'dt_16_60s' else 'dt_gt60s' end cadence, "
        "count(*) n from feat_steps where dt_s is not null group by 1").df()
    _cad = alt.Chart(_c).mark_bar().encode(
        x=alt.X("cadence:N", sort=["dt_1s", "dt_2s", "dt_3_5s", "dt_6_15s",
                                   "dt_16_60s", "dt_gt60s"],
                title="inter-point interval"),
        y=alt.Y("n:Q", title="steps")).properties(
        width=300, height=220, title="Sampling cadence (median 2 s)")
    mo.vstack([
        mo.md("## Kinematic texture (aggregates only)\n"
              "Speed classes separate the modes cleanly; turn innovations "
              "concentrate in the two near-zero 45° bins (smooth headings, "
              "not grid walks); the 1–2 s logging cadence is what makes "
              "staypoint detection and noise estimation possible."),
        mo.ui.altair_chart(_speed | _turn | _cad),
    ])
    return


@app.cell
def _(alt, con, mo):
    _rs = con.execute(
        "select mode_class, n_legs, n_users, "
        "round(median_circuity, 3) median_circuity, "
        "round(p90_circuity, 3) p90_circuity, "
        "round(median_straightness, 3) median_straightness, "
        "round(junctions_per_km, 2) junctions_per_km, "
        "round(share_left, 3) share_left "
        "from mart_route_shape order by n_legs desc").df()
    _seg = con.execute(
        "select seg_class, n_segments, n_users from mart_seg_hist").df()
    _ja = con.execute(
        "select ja_class, n_junctions, n_users from mart_junction_hist").df()
    _seg_order = ["seg_lt50", "seg_50_150", "seg_150_600", "seg_ge600"]
    _ja_order = ["ja_30_60", "ja_60_120", "ja_120_180"]
    _segc = alt.Chart(_seg).mark_bar().encode(
        x=alt.X("seg_class:N", sort=_seg_order,
                title="junction-to-junction segment length"),
        y=alt.Y("n_segments:Q", title="segments"),
        tooltip=["seg_class", "n_segments", "n_users"]).properties(
        width=300, height=220,
        title="Segment-length histogram (mart_seg_hist)")
    _jac = alt.Chart(_ja).mark_bar().encode(
        x=alt.X("ja_class:N", sort=_ja_order,
                title="junction turn angle (abs degrees)"),
        y=alt.Y("n_junctions:Q", title="junctions"),
        tooltip=["ja_class", "n_junctions", "n_users"]).properties(
        width=300, height=220,
        title="Junction-angle histogram (mart_junction_hist)")
    mo.vstack([
        mo.md("## Route shape (how trips bend)\n"
              "Legs are resampled to ≥25 m between kept points; a "
              "**junction** is a turn ≥30° on that stream; **segments** run "
              "junction-to-junction; **circuity** = path / net displacement. "
              "Per-mode circuity, straightness and junction density leave "
              "as the suppressed category-2 aggregates `route_shape`, "
              "`seg_hist` and `junction_hist` — and **synthetic trips are "
              "rendered from exactly these released statistics**: segment "
              "lengths and junction angles are drawn from the two "
              "histograms and per-mode circuity is targeted from "
              "`route_shape`, so generated paths bend like roads (segments "
              "+ junction turns) instead of flying as straight vectors. "
              "Walk is the most circuitous (1.99) and junction-dense (11.6 "
              "/km); car and rail are the straightest — the physics reads "
              "correctly."),
        mo.ui.table(_rs, selection=None),
        mo.ui.altair_chart(_segc | _jac),
    ])
    return


@app.cell
def _(alt, con, mo):
    _d = con.execute(
        "select round(latitude*50)/50 lat, round(longitude*50)/50 lon, "
        "count(*) n from stg_points group by 1, 2 "
        "having count(distinct user_id) >= 3").df()
    _chart = alt.Chart(_d).mark_rect().encode(
        x=alt.X("lon:Q", bin=alt.Bin(step=0.02), title="longitude"),
        y=alt.Y("lat:Q", bin=alt.Bin(step=0.02), title="latitude"),
        color=alt.Color("n:Q", scale=alt.Scale(type="log", scheme="magma"),
                        title="points"),
        tooltip=["lat", "lon", "n"]).properties(
        width=420, height=420,
        title="Corpus density heatbin (~2 km cells, ≥3 users per cell)")
    mo.vstack([
        mo.md("## Where the corpus lives (suppressed density, no "
              "trajectories)\n"
              "83.8% of points fall inside the Beijing bbox. Cells touched "
              "by fewer than 3 users are suppressed and **no individual "
              "trajectory is drawn** — this is the only spatial view the "
              "notebook permits itself. The synthetic flow never inherits "
              "this geography: the generator emits a neutral frame and the "
              "manipulate step relocates users onto Tokyo public geography."),
        mo.ui.altair_chart(_chart),
    ])
    return


@app.cell
def _(con, feature_collection):
    _cap = 150_000

    def _kept_sql(itv):
        return f"""
        with pts as (
            select user_id, trajectory_id, sequence_uuid, ts_s,
                   latitude, longitude, raw_mode,
                   min(ts_s) over (partition by user_id, trajectory_id,
                                   sequence_uuid) t0
            from stg_points
        ), marked as (
            select *,
                   row_number() over (
                       partition by user_id, trajectory_id, sequence_uuid,
                                    (ts_s - t0) // {itv}
                       order by ts_s) rn,
                   row_number() over (
                       partition by user_id, trajectory_id, sequence_uuid
                       order by ts_s desc) rl
            from pts
        )
        select * from marked where rn = 1 or rl = 1"""

    track_itv_s = 60
    _n = con.execute(
        f"select count(*) from ({_kept_sql(track_itv_s)})").fetchone()[0]
    while _n > _cap:
        track_itv_s = track_itv_s * _n // _cap + 1
        _n = con.execute(
            f"select count(*) from ({_kept_sql(track_itv_s)})").fetchone()[0]
    _legs = con.execute(f"""
        select user_id, trajectory_id, any_value(raw_mode) raw_mode,
               count(*) n,
               round((max(ts_s) - min(ts_s)) / 60.0, 1) mins,
               list([round(longitude, 5), round(latitude, 5)]
                    order by ts_s) coords
        from ({_kept_sql(track_itv_s)})
        group by user_id, trajectory_id, sequence_uuid
        having count(*) >= 2""").fetchall()
    tracks_fc = feature_collection([
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": _c},
         "properties": {"mode": _m, "user": _u, "trajectory": _t,
                        "points": _np, "minutes": _mins}}
        for _u, _t, _m, _np, _mins, _c in _legs])
    n_track_legs = len(_legs)
    n_track_coords = sum(_r[3] for _r in _legs)
    track_bounds = [round(_v, 4) for _v in con.execute(
        "select quantile_cont(longitude, 0.08), "
        "quantile_cont(latitude, 0.08), "
        "quantile_cont(longitude, 0.92), "
        "quantile_cont(latitude, 0.92) from stg_points").fetchone()]
    return n_track_coords, n_track_legs, track_bounds, track_itv_s, tracks_fc


@app.cell
def _(BASEMAPS, default_basemap_label, mo):
    basemap_pick = mo.ui.dropdown(
        options=list(BASEMAPS), value=default_basemap_label(),
        label="basemap")
    return (basemap_pick,)


@app.cell
def _(BASEMAPS, MapLibreMap, basemap_pick, mo, n_track_coords, n_track_legs,
      track_bounds, track_itv_s, tracks_fc):
    _palette = [("walk", "#f58518"), ("bike", "#54a24b"), ("bus", "#4c78a8"),
                ("car", "#e45756"), ("taxi", "#eeca3b"),
                ("subway", "#b279a2"), ("train", "#9d755d"),
                ("airplane", "#72b7b2"),
                ("other (run, motorcycle, boat)", "#bab0ac")]
    _legend = mo.Html("".join(
        f'<span style="display:inline-block;margin:0 12px 4px 0;'
        f'font:12px system-ui"><span style="display:inline-block;'
        f'width:12px;height:12px;background:{_c};border-radius:2px;'
        f'margin-right:4px;vertical-align:-1px"></span>{_m}</span>'
        for _m, _c in _palette))
    _map = mo.ui.anywidget(MapLibreMap(
        basemap=BASEMAPS[basemap_pick.value], bounds=track_bounds,
        height=620, layers={"tracks": tracks_fc}, visible=["tracks"]))
    mo.vstack([
        mo.md(
            f"## Trajectory map (ENCLAVE-ONLY — raw trajectories never "
            f"leave this boundary)\n"
            f"Every labelled leg as a line colored by its **raw GeoLife "
            f"modality label**. Rendered: {n_track_legs:,} legs / "
            f"{n_track_coords:,} points — each leg downsampled to one "
            f"point per {track_itv_s} s bucket (first and last point "
            f"always kept, legs with <2 kept points dropped) so the "
            f"browser is not handed the full 4.87M-point corpus. The "
            f"default view is the p08–p92 extent (the Beijing core, 84% of "
        f"points); world-spanning "
            f"outlier tracks still render if you zoom out. Unlike every "
            f"released artifact, this view shows individual users' "
            f"movement — it exists for enclave analysts only. The basemap "
            f"fetches tiles for the viewed area; choose "
            f"\"none (no network)\" for a fully offline map."),
        basemap_pick,
        _legend,
        _map,
    ])
    return


@app.cell
def _(alt, con, mo):
    _o = con.execute(
        "select origin, count(*) n, "
        "count(*) filter (arr_censored) n_arr_censored, "
        "count(*) filter (dep_censored) n_dep_censored "
        "from feat_staypoints group by 1").df()
    _dw = con.execute(
        "select dwell_class, count(*) n from feat_visits group by 1").df()
    _dw_order = ["dw_lt1h", "dw_1_3h", "dw_3_8h", "dw_8_14h", "dw_ge14h"]
    _dwc = alt.Chart(_dw).mark_bar().encode(
        x=alt.X("dwell_class:N", sort=_dw_order, title="dwell class (public)"),
        y=alt.Y("n:Q", title="staypoints")).properties(
        width=320, height=220, title="Dwell-class mix")
    mo.vstack([
        mo.md("## Staypoints: detected and implied\n"
              "5,669 staypoints (≥20 min within 200 m, Li et al.). The "
              "labelled subset records **movement**, so only 1,388 are "
              "classical `sp` detections; 4,281 are **implied** stays — the "
              "gap between one trajectory's end and the next one's start, "
              "departure-censored when the next start is elsewhere. "
              "Censoring is substantial (2,965 staypoints) and is carried "
              "explicitly rather than imputed."),
        mo.hstack([
            mo.ui.table(_o, selection=None),
            mo.ui.altair_chart(_dwc),
        ]),
    ])
    return


@app.cell
def _(alt, con, mo):
    _r = con.execute(
        "select role, count(*) n_anchors, count(distinct user_id) n_users "
        "from feat_anchors group by 1").df()
    _u = con.execute(
        "select count(*) n, count(*) filter (has_home) with_home, "
        "count(*) filter (has_work) with_work from mart_user_stats").fetchone()
    _chart = alt.Chart(_r).mark_bar().encode(
        x=alt.X("role:N", sort=["home", "work", "other"]),
        y=alt.Y("n_anchors:Q", title="anchors"),
        tooltip=["role", "n_anchors", "n_users"]).properties(
        width=300, height=220, title="Anchor roles (300 m grid cells)")
    mo.vstack([
        mo.md(f"## Anchors (ENCLAVE-ONLY locations)\n"
              f"2,013 anchors. `home` = the night-dwell (03:00) maximizer, "
              f"`work` = the weekday working-hours dwell maximizer (≥10 h "
              f"total). Of {_u[0]} users with anchors, **{_u[1]} have a home "
              f"anchor and {_u[2]} a work anchor** — a student-heavy corpus "
              f"where a stable 9-to-5 signature is the exception. Anchor "
              f"cells (`lat_c`/`lon_c`) are the most identifying artifact "
              f"in the warehouse and never leave; releases only ever see "
              f"role labels and relative structure."),
        mo.ui.altair_chart(_chart),
    ])
    return


@app.cell
def _(alt, con, mo):
    _m = con.execute(
        "select motif_class, day_type, count(*) n "
        "from feat_user_days group by 1, 2").df()
    _order = ["home_day", "commute", "errand_loop", "tour", "irregular"]
    _mix = alt.Chart(_m).mark_bar().encode(
        x=alt.X("motif_class:N", sort=_order, title="daily motif (public)"),
        y=alt.Y("n:Q", title="user-days"),
        tooltip=["motif_class", "day_type", "n"]).properties(
        width=300, height=220, title="Daily-motif mix (2,582 user-days)")
    _share = alt.Chart(_m).mark_bar().encode(
        x=alt.X("day_type:N", title=None),
        y=alt.Y("n:Q", stack="normalize", title="share of days"),
        color=alt.Color("motif_class:N", sort=_order),
        tooltip=["motif_class", "day_type", "n"]).properties(
        width=180, height=220, title="Weekday vs weekend")
    mo.vstack([
        mo.md("## Daily motifs (public thresholds; re-decided in Lean)\n"
              "Each observed user-day is classified: `commute` (home→work→"
              "home), `errand_loop` (short non-home stops, <3 h dwells), "
              "`tour` (≥3 distinct non-home anchors chained with ≤2 h "
              "gaps), `home_day`, else `irregular`. Movement-labelled "
              "recording makes `irregular` dominant (74%) and `commute` "
              "rare (29 days) — tours (283) out-number commutes 10:1, and "
              "the weekend mix shifts toward tours exactly as it should."),
        mo.ui.altair_chart(_mix | _share),
    ])
    return


@app.cell
def _(alt, con, mo):
    _t = con.execute(
        "select prev_role, next_role, n, round(p_next, 3) p_next, "
        "round(lift, 2) lift from mart_seq_transitions").df()
    _roles = ["home", "work", "other"]
    _heat = alt.Chart(_t).mark_rect().encode(
        x=alt.X("next_role:N", sort=_roles, title="to role"),
        y=alt.Y("prev_role:N", sort=_roles, title="from role"),
        color=alt.Color("lift:Q", scale=alt.Scale(scheme="blueorange",
                                                  domainMid=1),
                        title="lift"),
        tooltip=["prev_role", "next_role", "n", "p_next", "lift"]).properties(
        width=280, height=280,
        title="Role-transition matrix (color = lift vs independence)")
    _txt = alt.Chart(_t).mark_text(baseline="middle").encode(
        x=alt.X("next_role:N", sort=_roles),
        y=alt.Y("prev_role:N", sort=_roles),
        text="n:Q")
    mo.vstack([
        mo.md("## Role transitions (exact, enclave)\n"
              "5,163 sequence-ok transitions (observation gaps ≤6 h; 445 "
              "breaks excluded). Volume lives in `other→other`, but **lift** "
              "exposes the structure: `work→work` (2.17), `work→home` "
              "(1.47) and `home→work` (1.29) are the over-represented "
              "edges — the commuting skeleton — while `home→home` is "
              "under-represented (0.88). The DP release ships this grid as "
              "clipped noisy counts; the enclave sees it exactly."),
        mo.ui.altair_chart(_heat + _txt),
    ])
    return


@app.cell
def _(alt, con, mo):
    _c = con.execute(
        "select prev_mode, next_mode, n, round(p_next, 3) p_next, "
        "round(lift, 2) lift from mart_seq_chains").df()
    _order = ["walk", "bike", "bus", "car", "rail", "other"]
    _heat = alt.Chart(_c).mark_rect().encode(
        x=alt.X("next_mode:N", sort=_order, title="next leg mode"),
        y=alt.Y("prev_mode:N", sort=_order, title="previous leg mode"),
        color=alt.Color("n:Q", scale=alt.Scale(type="log", scheme="blues"),
                        title="chained legs"),
        tooltip=["prev_mode", "next_mode", "n", "p_next", "lift"]).properties(
        width=300, height=300, title="Mode-leg chains (consecutive legs)")
    mo.vstack([
        mo.md("## Mode chains\n"
              "Consecutive transport legs chain through **walk**: "
              "bus→walk (p 0.93) and walk→bus (lift 2.18) are the classic "
              "access/egress pattern, and same-mode persistence is strong "
              "for bike (lift 5.4). walk→walk is heavily suppressed (lift "
              "0.14) — the leg segmenter merges, as it should."),
        mo.ui.altair_chart(_heat),
    ])
    return


@app.cell
def _(alt, con, mo):
    _assoc = con.execute(
        "select feat_a, feat_b, round(chi2, 1) chi2, dof, n, "
        "round(cramers_v, 3) cramers_v from mart_feature_assoc "
        "order by cramers_v desc").df()
    _assoc["pair"] = _assoc["feat_a"] + " × " + _assoc["feat_b"]
    _bar = alt.Chart(_assoc).mark_bar().encode(
        x=alt.X("cramers_v:Q", title="Cramér's V"),
        y=alt.Y("pair:N", sort="-x", title=None),
        tooltip=["pair", "chi2", "dof", "n", "cramers_v"]).properties(
        width=480, height=180,
        title="Association strength (χ² recomputed exactly over ℚ in Lean)")
    mo.vstack([
        mo.md("## Feature associations\n"
              "Pre-declared categorical pairs, χ² + Cramér's V. The "
              "physical couplings lead — mode×distance (V 0.46) and "
              "mode×speed (V 0.43) — followed by role×dwell (home/work "
              "dwell long, errand dwell short) and the weaker but real "
              "role×arrival and motif×day-type structure. The same tables "
              "are re-derived in Lean from the exact counts, so the floats "
              "here are presentation only."),
        mo.ui.altair_chart(_bar),
        mo.ui.table(_assoc.drop(columns=["pair"]), selection=None),
    ])
    return


@app.cell
def _(alt, con, mo, pd):
    _n = con.execute(
        "select metric, value, n_staypoints, n_users "
        "from mart_noise_moments order by metric").df()
    _rows = []
    for _, _r in _n.iterrows():
        _tight = _r["metric"].startswith("tight_")
        _rows.append({"metric": _r["metric"].removeprefix("tight_"),
                      "subset": "tight (p95 ≤ 50 m)" if _tight else "all sp",
                      "value": _r["value"]})
    _d = pd.DataFrame([r for r in _rows if r["metric"] != "resid_lag1_mean"])
    _bar = alt.Chart(_d[~_d.metric.str.startswith("resid_lag1")]).mark_bar().encode(
        x=alt.X("metric:N", title=None),
        y=alt.Y("value:Q", title="meters"),
        color="subset:N", xOffset="subset:N").properties(
        width=340, height=220, title="GPS-noise scale: all vs tight subset")
    mo.vstack([
        mo.md("## GPS-noise moments (what the generator's wobble targets)\n"
              "Residuals of staypoint fixes around their centroid. Over all "
              "`sp` staypoints the median residual sd is 62.5 m with lag-1 "
              "autocorrelation 0.90 — dominated by multipath drift. The "
              "**tight subset** (residual p95 ≤ 50 m, 58 staypoints / 19 "
              "users) isolates true receiver noise: sd 23.2 m, lag-1 0.78. "
              "The released category-2 moments feed the synthetic AR(1) "
              "wobble; the staypoint coordinates themselves stay here."),
        mo.ui.altair_chart(_bar),
        mo.ui.table(_n, selection=None),
    ])
    return


@app.cell
def _(alt, con, mo):
    _dp = con.execute(
        "select 'motif' grid, cell, clipped_count, n_contributing_users, clip "
        "from mart_dp_motif "
        "union all select 'transitions', * from mart_dp_transitions "
        "union all select 'dwell', * from mart_dp_dwell "
        "union all select 'departure', * from mart_dp_departure "
        "union all select 'mode', * from mart_dp_mode "
        "union all select 'spd', * from mart_dp_spd "
        "union all select 'distance', * from mart_dp_distance").df()
    _chart = alt.Chart(_dp).mark_bar().encode(
        x=alt.X("cell:N", title=None),
        y=alt.Y("clipped_count:Q", title="clipped count"),
        tooltip=["grid", "cell", "clipped_count", "n_contributing_users",
                 "clip"]).properties(width=190, height=160).facet(
        facet=alt.Facet("grid:N", title=None), columns=4).resolve_scale(
        x="independent", y="independent")
    mo.vstack([
        mo.md("## DP inputs: clipped counts per grid (pre-noise, enclave)\n"
              "Each user contributes at most `clip = 10` across a grid "
              "(largest-remainder allocation), so one person moves any cell "
              "by ≤ 10 — the sensitivity the Lean mechanism proof consumes "
              "(`clippedSum_sensitivity`). With only **n = 64 users** the "
              "clip is deliberately small to keep the noise scale "
              "comparable to the dominant cell masses."),
        mo.ui.altair_chart(_chart),
    ])
    return


@app.cell
def _(ROOT, alt, con, json, mo, pd):
    _rel_p = ROOT / "lean" / "out" / "dp_release.json"
    _out = mo.md("*(no DP release yet — run `make train`)*")
    if _rel_p.exists():
        _rel = json.loads(_rel_p.read_text())
        _clip = {("motif", c): int(n) for c, n, _, _ in con.execute(
            "select * from mart_dp_motif").fetchall()}
        for _g, _mart in (("transitions", "mart_dp_transitions"),
                          ("dwell", "mart_dp_dwell"),
                          ("departure", "mart_dp_departure"),
                          ("mode", "mart_dp_mode")):
            for _c, _n, _, _ in con.execute(f"select * from {_mart}").fetchall():
                _clip[(_g, _c)] = int(_n)
        _rows = [{"grid": "motif", "cell": c["cell"],
                  "value": int(c["noised_count"]), "kind": "released (ε-DP)"}
                 for c in _rel["cells"]]
        _rows += [{"grid": h["name"], "cell": c["cell"],
                   "value": int(c["noised_count"]), "kind": "released (ε-DP)"}
                  for h in _rel.get("aux", []) for c in h["cells"]]
        _rows += [{"grid": g, "cell": c, "value": n,
                   "kind": "clipped (pre-noise, enclave)"}
                  for (g, c), n in _clip.items()]
        _chart = alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
            x=alt.X("cell:N", title=None),
            y=alt.Y("value:Q", title=None),
            color="kind:N", xOffset="kind:N",
            tooltip=["grid", "cell", "kind", "value"]).properties(
            width=230, height=170).facet(
            facet=alt.Facet("grid:N", title=None), columns=3).resolve_scale(
            x="independent", y="independent")
        _led = _rel["budget"]
        _out = mo.vstack([
            mo.md(f"## The ε-DP release (what actually leaves)\n"
                  f"Privacy unit: **one user** (n = 64 — small, so SNR is "
                  f"the honest question; compare bar pairs). The daily-motif "
                  f"histogram is the main release; role transitions, dwell, "
                  f"departure and mode mixes ride as aux histograms, "
                  f"{_led['n_cells']} + {_led['n_aux_cells']} cells in all. "
                  f"Ledger (exact fractions, no floats): "
                  f"**{_led['spent_num']}/{_led['spent_den']}** of "
                  f"{_led['total_num']}/{_led['total_den']}, exactly "
                  f"saturated — `lake exe release` exits 2 over budget."),
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
        _agree = _res.get("agreement", {})
        _out = mo.md(
            f"## Formal layer\n"
            f"Claims: **{_proved} proved**, {_assumed} assumed (trust "
            f"boundary stated explicitly). SQL↔Lean re-decisions: "
            f"**{_agree.get('n_agree', 0):,}/{_agree.get('n_checked', 0):,}** "
            f"agree ({_agree.get('n_disagree', 0)} disagreements) across "
            f"motif days, visit sequences, revisit anchors, long dwells and "
            f"every public binning (speed, dwell, departure, distance, "
            f"staypoints-per-day).")
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
        toolchain (Lean v4.34.1, Mathlib + **SampCert** for the verified DP
        noise). The library `lean/EnclaveGeo/` holds the definitions and
        proofs; the repo-root `lean/*.lean` files are the executables `lake`
        links against it:

        | File | What it formalizes |
        |---|---|
        | `EnclaveGeo/Params.lean` | the concept spec mirrored from dbt `vars:` + the raw-label→mode-class map (locked by `tests/test_spec_sync.py`), and the geography constants with `excl_tgt_disjoint`: the Beijing exclusion box and the Tokyo target box are provably disjoint |
        | `EnclaveGeo/Motifs.lean` | **decidable** public binnings (`dwellClass`, `hourBin`, `distClass`, `speedClass`, `spdBin`) and day predicates — `isCommuteDay`, `isTourDay`, `isErrandPair`, `IsLongDwell`, `IsRevisitAt` (≥5 distinct days in a 14-day window, exact `Nat` arithmetic, no floats) — with canonical-example and monotonicity theorems |
        | `EnclaveGeo/Sequences.lean` | `IsVisitSeqAt`: gap-bounded, chronologically ordered visit chains (the sequence-ok predicate the transition marts condition on) |
        | `EnclaveGeo/Stats.lean` | clipped per-user cell sums; `clippedSum_sensitivity` (add/remove one user moves a cell by ≤ clip) |
        | `EnclaveGeo/DPRelease.lean` | the release mechanism on SampCert discrete Laplace: pure-DP per cell and the composed multi-histogram budget |
        | `EnclaveGeo/Sampler.lean` | the verified categorical sampler; `sampleCellPMF_law`: P(i) = wᵢ/Σw, exactly |
        | `EnclaveGeo/Assoc.lean` | χ² recomputed over ℚ; `chiSq_nonneg`, independence ⇒ χ² = 0 |
        | `EnclaveGeo/GenSpec.lean` | the generator's typed spec + smart constructors with validity **by construction**: `mkStays_times_sorted` (arrive/leave strictly increasing), `mkStays_isStay` (every planted stay clears the staypoint threshold), `mkStays_visitSeq` (planted days re-ingest as sequence-ok chains) |
        | `EnclaveGeo/Motion.lean` | public physics and rendering: speed caps per mode, the kernel-checked calendar, wobble-box containment, trajectory/uuid id shapes |
        | `Release.lean` / `Checks.lean` / `Generate.lean` / `Sample.lean` / `Catsample.lean` | executables: ε-DP release (exact-fraction ledger, exit 2 over budget), SQL↔Lean re-deciding, the standalone synthetic trajectory generator, sampler tools |

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
        _files = (sorted((ROOT / "lean" / "EnclaveGeo").glob("*.lean"))
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
