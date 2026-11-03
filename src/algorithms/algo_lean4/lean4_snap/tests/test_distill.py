"""Distillation: the genspec is pure post-processing of released values —
every weight equals a released value or released + a DECLARED public prior."""
import hashlib
import json

import pytest

DEG_BINS = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16", "deg_17_32",
            "deg_33_64", "deg_65p"]
CC = ["cc_zero", "cc_0_20", "cc_20_40", "cc_40_60", "cc_60_80", "cc_80_99",
      "cc_one"]
TRI = ["tri_0", "tri_1_10", "tri_11_100", "tri_101_1000", "tri_1001_10000",
       "tri_10001p"]
RICH = [f"rich_{i}" for i in range(8)] + ["rich_8p"]
CIRC = ["circ_0", "circ_1", "circ_2", "circ_3", "circ_4p"]
CSIZE = ["csize_lt25", "csize_25_50", "csize_51_100", "csize_101_200",
         "csize_201_400", "csize_gt400"]
CATS = ["birthday", "education", "first_name", "gender", "hometown",
        "languages", "last_name", "locale", "location", "work", "other"]
RANKS = [f"c{i}" for i in range(1, 9)] + ["rest"]
MOTIFS = ["motif_hub", "motif_leaf", "motif_broker", "motif_bridge",
          "motif_butterfly", "motif_core"]
AFFIL_CATS = ["birthday", "education", "gender", "hometown", "languages",
              "last_name", "locale", "location", "work"]
AFFIL_BINS = ["g_2_4", "g_5_8", "g_9_16", "g_17p"]


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


def test_degree_equals_release(rel_and_spec):
    rel, gs, _ = rel_and_spec
    released = {c["cell"]: max(0, int(c["noised_count"])) for c in rel["cells"]}
    got = _cells(gs, "degree")
    assert set(got) == set(DEG_BINS)
    for name, w in got.items():
        assert w == released[name], name


@pytest.mark.parametrize("key,aux_name,names", [
    ("clustering", "clustering", CC),
    ("triangles", "triangles", TRI),
    ("richness", "richness", RICH),
    ("circles", "circles", CIRC),
    ("motifs", "motifs", MOTIFS),
])
def test_marginals_equal_release(rel_and_spec, key, aux_name, names):
    _, gs, aux = rel_and_spec
    got = _cells(gs, key)
    assert set(got) == set(names), key
    for name, w in got.items():
        assert w == max(0, aux[aux_name].get(name, 0)), (key, name)


def test_csize_is_release_over_public_midpoint(rel_and_spec):
    """The community-size weights equal the released NODE-weighted counts
    divided by the PUBLIC bin midpoint (declared post-processing prior
    csize_node_weighted)."""
    _, gs, aux = rel_and_spec
    got = _cells(gs, "community_sizes")
    assert set(got) == set(CSIZE)
    for name, w in got.items():
        lo, hi = gs["csize_ranges"][name]
        mid = max(1, (lo + hi) // 2)
        rel_w = max(0, aux["community_sizes"].get(name, 0))
        assert w == (rel_w + mid // 2) // mid, name


def test_category_cells_are_released_minus_prefix(rel_and_spec):
    """Category weights equal the released cat_* cells; the name change
    (prefix strip) is declared post-processing."""
    _, gs, aux = rel_and_spec
    got = _cells(gs, "category")
    assert set(got) == set(CATS)
    for name, w in got.items():
        assert w == max(0, aux["category"].get(f"cat_{name}", 0)), name


def test_mixing_rows_are_symmetric_expansion(rel_and_spec):
    """Mixing row[a][b] must equal the released unordered cell (min,max)."""
    _, gs, aux = rel_and_spec
    mix = aux["mixing"]
    rows = {r["given"]: {c["name"]: c["w"] for c in r["cells"]}
            for r in gs["mixing"]["rows"]}
    assert set(rows) == set(RANKS)
    for a in RANKS:
        for b in RANKS:
            ia, ib = RANKS.index(a), RANKS.index(b)
            key = f"{a}__{b}" if ia <= ib else f"{b}__{a}"
            assert rows[a][b] == max(0, mix.get(key, 0)), (a, b)
    # symmetry is a theorem of the expansion
    for a in RANKS:
        for b in RANKS:
            assert rows[a][b] == rows[b][a], (a, b)


def test_closure_rows_equal_release(rel_and_spec):
    _, gs, aux = rel_and_spec
    clo = aux["closure"]
    rows = {r["given"]: {c["name"]: c["w"] for c in r["cells"]}
            for r in gs["closure"]["rows"]}
    assert set(rows) == set(DEG_BINS)
    for d in DEG_BINS:
        assert rows[d]["closed"] == max(0, clo.get(f"clo_{d}__closed", 0)), d
        assert rows[d]["open"] == max(0, clo.get(f"clo_{d}__open", 0)), d


def test_ranges_are_public_spec(rel_and_spec):
    from spec import load_spec
    _, gs, _ = rel_and_spec
    s = load_spec()
    assert gs["deg_ranges"]["deg_3_4"] == [int(s["deg_e3"]), int(s["deg_e5"]) - 1]
    assert gs["deg_ranges"]["deg_65p"][0] == int(s["deg_e65"])
    assert gs["csize_ranges"]["csize_25_50"] == [int(s["csize_e25"]),
                                                 int(s["csize_e51"]) - 1]
    assert gs["ego_count"] == 10


def test_priors_are_declared(rel_and_spec):
    _, gs, _ = rel_and_spec
    priors = {p["name"]: p for p in gs["priors"]}
    names = ["clamp_at_zero", "bin_interior_uniform", "closure_trials",
             "community_scoped_features", "csize_node_weighted"]
    if "affiliation" in gs:
        names += ["hub_full_deficit", "affil_member_weighted", "affil_window",
                  "affil_closure_bias", "affil_stochastic_round",
                  "affil_group_sample",
                  "affil_density_fill",
                  "affil_suppressed_bins_skipped"]
    for name in names:
        assert name in priors, name
        assert priors[name]["derived_from_data"] is False, name


@pytest.fixture(scope="module")
def affil(rel_and_spec, root):
    """(genspec affiliation block, DP affil_sizes aux, released density
    rows) — skips on a pre-affiliation release."""
    _, gs, aux = rel_and_spec
    if "affiliation" not in gs:
        pytest.skip("genspec predates the affiliation release")
    coh_p = root / "lean" / "out" / "affil_cohesion.json"
    if not coh_p.exists():
        pytest.skip("released affiliation cohesion rows missing")
    coh = json.loads(coh_p.read_text())
    dens = {(r["category"], r["size_bin"]): int(r["density_pct"])
            for r in coh["rows"]}
    return gs["affiliation"], aux["affil_sizes"], dens


def test_affiliation_sizes_equal_release(affil):
    """Size weights equal the DP affil_sizes cells divided by the PUBLIC bin
    midpoint (declared prior affil_member_weighted) — except bins whose
    cohesion cell was suppressed, which enter as 0."""
    block, asz, dens = affil
    rows = {r["given"]: {c["name"]: c["w"] for c in r["cells"]}
            for r in block["size_rows"]}
    assert set(rows) == set(AFFIL_CATS)
    for c in AFFIL_CATS:
        assert set(rows[c]) == set(AFFIL_BINS)
        for b, w in rows[c].items():
            if (c, b) in dens:
                lo, hi = block["group_ranges"].get(f"{c}/{b}", block["group_ranges"][b])
                mid = max(1, (lo + hi) // 2)
                rel_w = max(0, asz.get(f"aff_{c}_{b}", 0))
                assert w == (rel_w + mid // 2) // mid, (c, b)
            else:
                assert w == 0, (c, b)


def test_affiliation_densities_are_released(affil):
    """Every density weight equals the suppression-released integer percent
    (clamped to [0, 100]); suppressed cells never appear."""
    block, _, dens = affil
    assert block["density_den"] == 100
    rows = {r["given"]: {c["name"]: c["w"] for c in r["cells"]}
            for r in block["density_rows"]}
    assert set(rows) == set(AFFIL_CATS)
    for c in AFFIL_CATS:
        for b, w in rows[c].items():
            assert (c, b) in dens, (c, b)
            assert w == min(100, max(0, dens[(c, b)])), (c, b)
    for (c, b), _ in dens.items():
        assert b in rows[c], (c, b)


def test_fidelity_report_honest(root):
    p = root / "models_out" / "fidelity.json"
    if not p.exists():
        pytest.skip("run `make distill` first")
    rep = json.loads(p.read_text())
    if "degree_tv_released_vs_exact" not in rep:
        pytest.skip("fidelity computed without the warehouse")
    assert rep["accepted"] == (rep["degree_tv_released_vs_exact"] <= 0.15), rep
