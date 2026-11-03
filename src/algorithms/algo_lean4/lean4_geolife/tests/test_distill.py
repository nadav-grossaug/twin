"""Distillation: the genspec is pure post-processing of released values —
every DP-derived weight equals a released noised count (clamped at 0) or
released + a DECLARED public prior. The shape is the GenSpec.lean contract
(`GenSpec.parse`): keys motif/transitions/dwell/departure/mode/turn/dist_mix/
kinematics/noise/window/dwell_ranges_s/dep_ranges_h/dist_ranges_m, weights
non-negative ints. Skips cleanly until `make train` (distill) has run."""
import hashlib
import json

import pytest

from spec import (DEP_BINS, DIST_CLASSES, DWELL_CLASSES, MODE_CLASSES,
                  MOTIF_CLASSES, ROLES, TURN_BINS, load_spec)


@pytest.fixture(scope="module")
def rel_and_spec(root):
    rel_p = root / "lean" / "out" / "dp_release.json"
    gs_p = root / "lean" / "out" / "genspec.json"
    if not rel_p.exists() or not gs_p.exists():
        pytest.skip("run `make train` first")
    rel = json.loads(rel_p.read_text())
    gs = json.loads(gs_p.read_text())
    if gs["source"]["dp_release_sha256"] != hashlib.sha256(
            rel_p.read_bytes()).hexdigest():
        pytest.skip("genspec predates the current (non-reproducible) release — "
                    "run `make distill` after the lean release")
    aux = {h["name"]: {c["cell"]: int(c["noised_count"]) for c in h["cells"]}
           for h in rel["aux"]}
    return rel, gs, aux


def _cells(gs, key):
    return {c["name"]: c["w"] for c in gs[key]["cells"]}


# --------------------------------------------------------- shape contract
def test_genspec_matches_lean_contract(rel_and_spec):
    """Everything GenSpec.parse reads must be present and typed."""
    _, gs, _ = rel_and_spec
    assert isinstance(gs["spec_version"], str)
    assert isinstance(gs["n_users_hint"], int) and gs["n_users_hint"] >= 0
    for key in ("motif", "transitions", "dwell", "departure", "mode",
                "turn", "dist_mix", "kinematics", "noise", "window",
                "dwell_ranges_s", "dep_ranges_h", "dist_ranges_m"):
        assert key in gs, key
    for k in gs["kinematics"]:
        assert k["mode"] in MODE_CLASSES
        assert isinstance(k["speed_mmps"], int) and k["speed_mmps"] >= 0
        assert isinstance(k["dt_ds"], int) and k["dt_ds"] >= 0
        assert isinstance(k.get("path_m", 0), int) and k.get("path_m", 0) >= 0
    noise = gs["noise"]
    assert isinstance(noise["sd_mm"], int) and noise["sd_mm"] >= 0
    assert isinstance(noise["lag1_pct"], int) and 0 <= noise["lag1_pct"] <= 100


def test_weights_are_nonnegative_ints(rel_and_spec):
    _, gs, _ = rel_and_spec
    for key in ("motif", "dwell", "departure", "mode", "turn", "dist_mix"):
        for c in gs[key]["cells"]:
            assert isinstance(c["w"], int) and c["w"] >= 0, (key, c)
    for row in gs["transitions"]["rows"]:
        for c in row["cells"]:
            assert isinstance(c["w"], int) and c["w"] >= 0, ("transitions", c)


def test_grids_are_the_public_taxonomies(rel_and_spec):
    _, gs, _ = rel_and_spec
    assert [c["name"] for c in gs["motif"]["cells"]] == MOTIF_CLASSES
    assert [c["name"] for c in gs["dwell"]["cells"]] == DWELL_CLASSES
    assert [c["name"] for c in gs["departure"]["cells"]] == DEP_BINS
    assert [c["name"] for c in gs["mode"]["cells"]] == MODE_CLASSES
    assert [c["name"] for c in gs["turn"]["cells"]] == TURN_BINS
    assert {c["name"] for c in gs["dist_mix"]["cells"]} == set(DIST_CLASSES)
    assert [r["given"] for r in gs["transitions"]["rows"]] == ROLES
    for row in gs["transitions"]["rows"]:
        assert [c["name"] for c in row["cells"]] == ROLES


# -------------------------------------------------------- DP provenance
def test_motif_equals_release(rel_and_spec):
    rel, gs, _ = rel_and_spec
    priors = {p["name"] for p in gs.get("priors", [])}
    released = {c["cell"]: max(0, int(c["noised_count"])) for c in rel["cells"]}
    for name, w in _cells(gs, "motif").items():
        want = released[name]
        if "motif_floor" in priors:    # declared: zero cells floored at 1
            want = max(1, want)
        assert w == want, name


@pytest.mark.parametrize("key,aux_name,names", [
    ("dwell", "dwell", DWELL_CLASSES),
    ("departure", "departure", DEP_BINS),
    ("mode", "mode", MODE_CLASSES),
])
def test_marginals_equal_release(rel_and_spec, key, aux_name, names):
    _, gs, aux = rel_and_spec
    got = _cells(gs, key)
    assert set(got) == set(names), key
    for name, w in got.items():
        assert w == max(0, aux[aux_name].get(name, 0)), (key, name)


def test_transition_rows_follow_declared_priors(rel_and_spec):
    """Each transition weight is the clamped released count, plus EXACTLY the
    prior the genspec DECLARES (derived_from_data: false) — nothing else.
    The declared home_return_boost rule: on every row given g != 'home', the
    'home' cell gets max(1, row_sum // 2) added."""
    _, gs, aux = rel_and_spec
    trans = aux["transitions"]
    priors = {p["name"]: p for p in gs.get("priors", [])}
    for p in priors.values():
        assert p["derived_from_data"] is False, p
    boosted = "home_return_boost" in priors
    for row in gs["transitions"]["rows"]:
        g = row["given"]
        raw = {n: max(0, trans.get(f"{g}__{n}", 0)) for n in ROLES}
        boost = max(1, sum(raw.values()) // 2)
        for cell in row["cells"]:
            n = cell["name"]
            want = raw[n] + (boost if boosted and n == "home" and g != "home"
                             else 0)
            assert cell["w"] == want, (g, n, cell["w"], want)


# ---------------------------------------------------- public spec mirrors
def test_window_and_ranges_are_public_spec(rel_and_spec):
    _, gs, _ = rel_and_spec
    s = load_spec()
    assert gs["window"]["start_s"] == int(s["window_start_s"])
    assert gs["window"]["end_s"] == int(s["window_end_s"])
    dw = gs["dwell_ranges_s"]
    assert set(dw) == set(DWELL_CLASSES)
    assert dw["dw_lt1h"][1] in (int(s["dw_e1_s"]), int(s["dw_e1_s"]) - 1)
    dep = gs["dep_ranges_h"]
    assert set(dep) == set(DEP_BINS)
    assert dep["dep_night"][1] == int(s["dep_night_max_h"])
    dist = gs["dist_ranges_m"]
    assert set(dist) == set(DIST_CLASSES)
    assert dist["dist_lt1k"][1] in (int(s["dist_e1_m"]), int(s["dist_e1_m"]) - 1)
    for ranges in (dw, dep, dist):
        for name, (lo, hi) in ranges.items():
            assert isinstance(lo, int) and isinstance(hi, int) and lo <= hi, name


# ------------------------------------------------------- route shape (new)
# The route-shape genspec keys are pure post-processing of the three new
# category-2 artifacts (route_shape / seg_hist / junction_hist). An older
# release lacks them, in which case distill's DECLARED fallback priors stand
# in and genspec_meta records no source for the artifact.

SEG_CLASSES = ["seg_lt50", "seg_50_150", "seg_150_600", "seg_ge600"]
JA_CLASSES = ["ja_30_60", "ja_60_120", "ja_120_180"]


@pytest.fixture(scope="module")
def route_genspec(rel_and_spec, root):
    _, gs, _ = rel_and_spec
    if not all(k in gs for k in ("seg_hist", "junction_hist", "routes")):
        pytest.skip("genspec predates the route-shape layer — "
                    "run `make train` (distill after a fresh release) first")
    meta_p = root / "lean" / "out" / "genspec_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    recorded = meta.get("source", {}).get("artifacts", {})
    return gs, recorded


def _released_rows(name: str):
    """The rows distill post-processed: the release distill SELECTED (export
    CSV, root CSV, or the policy-filtered mart) — exactly distill's own
    resolution, never blindly the latest export."""
    import distill
    try:
        rows, _src = distill._released_artifact(name, distill._select_release()[1])
    except (KeyError, PermissionError):
        pytest.skip(f"release policy does not (yet) carry artifact {name!r} "
                    f"— update config/release_policy.yaml / run `make train`")
    if rows is None:
        pytest.skip(f"no released {name} rows on disk to compare against — "
                    f"run `make train` first")
    return rows


def test_seg_hist_matches_release(route_genspec):
    import distill
    gs, recorded = route_genspec
    cells = gs["seg_hist"]["cells"]
    assert [c["name"] for c in cells] == SEG_CLASSES
    if "seg_hist" not in recorded:   # fallback tolerance per the distill meta
        assert [c["w"] for c in cells] == distill.FALLBACK_SEG_W
        return
    want = {r["seg_class"]: max(0, int(r["n_segments"]))
            for r in _released_rows("seg_hist")}
    for c in cells:
        assert c["w"] == max(0, int(want.get(c["name"], 0))), c


def test_junction_hist_matches_release(route_genspec):
    import distill
    gs, recorded = route_genspec
    cells = gs["junction_hist"]["cells"]
    assert [c["name"] for c in cells] == JA_CLASSES
    if "junction_hist" not in recorded:
        assert [c["w"] for c in cells] == distill.FALLBACK_JA_W
        return
    want = {r["ja_class"]: max(0, int(r["n_junctions"]))
            for r in _released_rows("junction_hist")}
    for c in cells:
        assert c["w"] == max(0, int(want.get(c["name"], 0))), c


def test_routes_match_release(route_genspec):
    """routes = {share_left_pct, circ_pm: [{mode, circ_pm, ...}]} — the
    GenSpec.lean parse contract — with circ_pm = int(median_circuity * 1000)
    and share_left_pct the n_legs-weighted released share (clamped 35..65)."""
    import distill
    gs, recorded = route_genspec
    routes = gs["routes"]
    assert isinstance(routes["share_left_pct"], int)
    assert 0 <= routes["share_left_pct"] <= 100
    by_mode = {}
    for c in routes["circ_pm"]:
        assert isinstance(c["mode"], str) and c["mode"] in MODE_CLASSES
        assert isinstance(c["circ_pm"], int) and c["circ_pm"] >= 0
        by_mode[c["mode"]] = c
    if "route_shape" not in recorded:
        assert routes["share_left_pct"] == distill.FALLBACK_SHARE_LEFT_PCT
        assert set(by_mode) == set(MODE_CLASSES)
        assert all(c["circ_pm"] == distill.FALLBACK_CIRC_PM
                   for c in by_mode.values())
        return
    rows = [r for r in _released_rows("route_shape")
            if r.get("median_circuity") not in (None, "")]
    assert rows, "released route_shape artifact is empty"
    legs_total = sum(max(1, int(r["n_legs"])) for r in rows)
    share_mean = sum(float(r["share_left"]) * max(1, int(r["n_legs"]))
                     for r in rows) / legs_total
    assert routes["share_left_pct"] == min(65, max(35, round(100 * share_mean)))
    for r in rows:
        c = by_mode[r["mode_class"]]
        assert c["circ_pm"] == max(
            0, int(float(r["median_circuity"]) * 1000)), r["mode_class"]
        if "junc_pkm_c" in c:
            assert c["junc_pkm_c"] == max(
                0, int(float(r["junctions_per_km"]) * 100)), r["mode_class"]


# ------------------------------------------ release selection (stale-export trap)
@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A fake project: lean/out (live release), release/<id> (export), DB."""
    import distill
    out = tmp_path / "lean" / "out"
    out.mkdir(parents=True)
    exp = tmp_path / "release" / "20260101T000000Z"
    exp.mkdir(parents=True)
    db = tmp_path / "artifacts" / "enclave.duckdb"
    db.parent.mkdir()
    monkeypatch.setattr(distill, "ROOT", tmp_path)
    monkeypatch.setattr(distill, "OUT", out)
    monkeypatch.setattr(distill, "DB", db)
    monkeypatch.delenv("RELEASE_DIR", raising=False)
    return distill, out, exp, db


def test_select_release_prefers_newer_live_release(layout):
    distill, out, exp, db = layout
    (exp / "dp_release.json").write_text('{"old": 1}')
    (out / "dp_release.json").write_text('{"new": 1}')
    db.write_text("x")
    rel_p, rel_dir, src = distill._select_release()
    assert rel_p == out / "dp_release.json"
    assert rel_dir is None            # category-2 from marts, not stale CSVs
    assert "live" in src


def test_select_release_uses_export_when_identical(layout):
    distill, out, exp, db = layout
    (exp / "dp_release.json").write_text('{"same": 1}')
    (out / "dp_release.json").write_text('{"same": 1}')
    db.write_text("x")
    rel_p, rel_dir, _ = distill._select_release()
    assert rel_p == exp / "dp_release.json" and rel_dir == exp


def test_select_release_bundle_without_db_uses_export(layout):
    distill, out, exp, _db = layout
    (exp / "dp_release.json").write_text('{"old": 1}')
    (out / "dp_release.json").write_text('{"new": 1}')
    rel_p, rel_dir, _ = distill._select_release()
    assert rel_p == exp / "dp_release.json" and rel_dir == exp


def test_select_release_explicit_release_dir_wins(layout, monkeypatch):
    distill, out, exp, db = layout
    (exp / "dp_release.json").write_text('{"old": 1}')
    (out / "dp_release.json").write_text('{"new": 1}')
    db.write_text("x")
    monkeypatch.setenv("RELEASE_DIR", str(exp))
    rel_p, rel_dir, _ = distill._select_release()
    assert rel_p == exp / "dp_release.json" and rel_dir == exp
