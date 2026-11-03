"""Different-dataset generality test (plan section 9): the method must run unchanged on a toy graph
whose schema shares nothing with SNAP — different attribute names, domain sizes, an ordered attribute
and a missing label that is not ``"none"``.

The full inside -> transfer -> outside flow runs in-process: ``fit`` builds the artifact, it is saved
to disk, loaded back, validated, and the generator samples a synthetic graph from the loaded artifact
alone. Once with noise (eps = 4) and once without (eps = inf).

Note: the transfer contract (``AttributedGraph`` / ``validate``) requires every domain to carry
explicit ``"other"`` and ``"none"`` labels, so those two appear in each toy domain; the *missing*
label of ``shape`` is ``"unknown"``, exercising a missing label different from ``"none"``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graph_aim.eval.coupling import coupling_metrics
from graph_aim.inside.fit import fit
from graph_aim.outside import generate as gen
from graph_aim.outside.generate import generate
from graph_aim.transfer.artifact import load, save, validate
from graph_aim.transfer.types import AttributedGraph

# Nothing here resembles the SNAP schema: names, sizes and order are all different.
DOMAINS = {
    "color": ["red", "blue", "other", "none"],  # 4 values; degree depends on it
    "shape": ["circle", "square", "unknown", "other", "none"],  # 5 values; missing label "unknown"
    "size_class": ["xs", "s", "m", "other", "none"],  # 5 values; ordered
}
MISSING = {"shape": "unknown"}  # not "none"
ORDERED = {"size_class": True}
MEAN_DEGREE_BY_COLOR = np.array([2.0, 9.0, 4.0, 4.0])  # red low, blue high
LO, HI = DOMAINS["color"].index("red"), DOMAINS["color"].index("blue")


def toy_graph(n: int = 300, seed: int = 42) -> AttributedGraph:
    """Deterministic toy graph: expected degree is set by ``color`` (Chung-Lu with weights 2..9)."""
    rng = np.random.default_rng(seed)
    attrs = pd.DataFrame(
        {
            "color": rng.choice(4, size=n, p=[0.40, 0.35, 0.15, 0.10]),
            "shape": rng.choice(5, size=n, p=[0.40, 0.30, 0.15, 0.10, 0.05]),
            "size_class": rng.choice(5, size=n, p=[0.30, 0.30, 0.25, 0.10, 0.05]),
        }
    ).astype(np.int64)
    w = MEAN_DEGREE_BY_COLOR[attrs["color"].to_numpy()]
    iu, iv = np.triu_indices(n, 1)
    p = np.minimum(w[iu] * w[iv] / w.sum(), 1.0)
    keep = rng.random(len(p)) < p
    edges = np.stack([iu[keep], iv[keep]], axis=1).astype(np.int64)
    return AttributedGraph(n, edges, attrs, {c: list(v) for c, v in DOMAINS.items()})


def toy_config(noise: bool) -> dict:
    """A complete public config written for the toy schema; no SNAP file is read."""
    return {
        "D": 12,
        "salt": 123,
        "max_degree": 40,
        "caps": {"clustering_by_degree": 12, "joint_degree": None, "attr_edge_pair": 8},
        "hist": {"exact_upto": 8, "growth": 1.5, "threshold": 1.0, "spread_power": 1.0},
        "cluster_bin_edges": [0, 2, 4, 8, 41],
        "cluster_min_degree": 4,
        "cluster_counts_from_hist": True,
        "cluster_clip": 1.0,
        "joint_bin_edges": [0, 1, 3, 6, 41],
        "n_bins": 8,
        "delta": 1e-6,
        "epsilon_edge": 4.0,
        "epsilon_attr": 4.0,
        "shares": {
            "degree_hist": 0.30,
            "clustering_by_degree": 0.28,
            "joint_degree": 0.22,
            "attr_1way": 0.06,
            "attr_2way": 0.44,
            "attr_edge_pair": 0.46,
            "attr_degree": 0.20,
        },
        "attr_degree_bin_edges": None,
        "attr_degree_attrs": None,
        "noise": noise,
        "homophily_attrs": None,  # first two attributes, whatever they are called
    }


@pytest.fixture(scope="module")
def toy() -> AttributedGraph:
    return toy_graph()


def _fit_save_load(toy: AttributedGraph, noise: bool, tmp_path_factory) -> object:
    art = fit(toy, toy_config(noise), seed=11)
    path = save(art, tmp_path_factory.mktemp("artifacts") / f"toy_{'noise' if noise else 'exact'}_seed11")
    loaded = load(path)
    assert validate(loaded) == []
    return loaded


@pytest.fixture(scope="module")
def noisy_artifact(toy, tmp_path_factory):
    return _fit_save_load(toy, True, tmp_path_factory)


@pytest.fixture(scope="module")
def exact_artifact(toy, tmp_path_factory):
    return _fit_save_load(toy, False, tmp_path_factory)


def _artifact(request, which: str):
    return request.getfixturevalue(f"{which}_artifact")


def _assign(art, seed: int = 0):
    """Attribute sampling + pool + private degree assignment, exactly as ``generate`` does it."""
    n = art.schema.n
    rng = np.random.default_rng(seed)
    attrs = gen._sample_attributes(art, n, rng)
    tabs = gen._attr_degree_tables(art, attrs)
    assert tabs is not None  # the toy fit released attr_degree tables and they are usable
    h = gen._clean(art.measurements["degree_hist"].values)
    exact = h / h.sum() * n
    pool = np.minimum(np.searchsorted(np.cumsum(exact), rng.random() + np.arange(n), side="left"), len(h) - 1)
    deg, info = gen._assign_degrees_by_attributes(pool, attrs, tabs, rng)
    return attrs, pool, deg, info


# ---------------------------------------------------------------------------- artifact round trip
@pytest.mark.parametrize("which", ["noisy", "exact"])
def test_artifact_round_trip_carries_toy_schema(request, which):
    art = _artifact(request, which)
    assert art.schema.domains == DOMAINS
    assert art.schema.homophily_attrs == ["color", "shape"]
    names = [k for k in art.measurements if k.startswith("attr_degree:")]
    assert names == [f"attr_degree:{c}" for c in DOMAINS]
    for c, dom in DOMAINS.items():
        m = art.measurements[f"attr_degree:{c}"]
        assert m.values.shape[0] == len(dom) and np.all(m.values >= 0)


# ------------------------------------------------------------------------------------- generation
@pytest.mark.parametrize("which", ["noisy", "exact"])
def test_generate_succeeds_with_toy_domains(request, toy, which):
    art = _artifact(request, which)
    g = generate(art, seed=5, rewire_rounds=2)
    assert g.n == toy.n
    assert g.domains == DOMAINS  # synthetic attribute domains equal the toy domains
    e = g.edges
    assert e.shape[1] == 2 and (e[:, 0] < e[:, 1]).all()
    assert len(np.unique(e[:, 0] * g.n + e[:, 1])) == len(e)  # simple graph
    for c, dom in DOMAINS.items():
        codes = g.attrs[c].to_numpy()
        assert codes.min() >= 0 and codes.max() < len(dom)
    g2 = generate(art, seed=5, rewire_rounds=2)
    assert np.array_equal(g.edges, g2.edges) and g.attrs.equals(g2.attrs)  # deterministic per seed


@pytest.mark.parametrize("which", ["noisy", "exact"])
def test_degree_histogram_exactly_matches_released_pool(request, which):
    art = _artifact(request, which)
    attrs, pool, deg, info = _assign(art, seed=0)
    assert np.array_equal(np.sort(deg), np.sort(pool))  # the pool is a permutation of the degrees
    assert info["attrs"] == list(DOMAINS)  # all three toy tables were used
    assert np.isfinite(info["residual"])


def test_exact_artifact_recovers_coupling_direction(toy, exact_artifact):
    """At eps = inf the value with the high real mean degree gets the high synthetic mean degree."""
    deg_real = np.bincount(toy.edges.ravel(), minlength=toy.n)
    color = toy.attrs["color"].to_numpy()
    real_hi, real_lo = deg_real[color == HI].mean(), deg_real[color == LO].mean()
    assert real_hi > real_lo + 2  # the toy construction did couple degree to color

    # direction in the assigned degrees (before structure generation)
    attrs, _, deg, _ = _assign(exact_artifact, seed=0)
    c = attrs["color"].to_numpy()
    assert deg[c == HI].mean() > deg[c == LO].mean() + 2

    # and in the realized degrees of the generated graph
    g = generate(exact_artifact, seed=7, rewire_rounds=2)
    d = np.bincount(g.edges.ravel(), minlength=g.n)
    cs = g.attrs["color"].to_numpy()
    assert (cs == HI).any() and (cs == LO).any()
    assert d[cs == HI].mean() > d[cs == LO].mean() + 1


# ------------------------------------------------------------------------------------ eval metrics
def test_coupling_metrics_run_on_toy_pair(toy, exact_artifact):
    g = generate(exact_artifact, seed=7, rewire_rounds=2)
    edges = exact_artifact.measurements["attr_degree:color"].params["bin_edges"]
    out = coupling_metrics(
        toy, g, degree_bin_edges=edges, missing=MISSING, ordered=ORDERED, exclude_hubs=2
    )
    assert set(out["attributes"]) == set(DOMAINS)
    for col, m in out["attributes"].items():
        # ordered-attribute distance only for the ordered column
        assert ("mean_abs_bin_distance_real" in m) == (col == "size_class")
        assert np.isfinite(m["deg_by_value_tvd"])
        assert "deg_by_value_tvd_nohub" in m  # hub exclusion ran
    sh = out["attributes"]["shape"]
    assert 0.0 < sh["observed_pair_share_real"] < 1.0  # the custom missing label was excluded
    col = out["attributes"]["color"]
    assert np.isfinite(col["eta2_real"]) and col["eta2_real"] > 0.1  # real coupling is visible
    assert col["deg_by_value_tvd"] < 0.5  # eps = inf: well under the independent baseline (~1)
    tab = out["mean_degree_by_value"]
    row = tab[(tab["attribute"] == "color") & (tab["value"] == "blue")].iloc[0]
    low = tab[(tab["attribute"] == "color") & (tab["value"] == "red")].iloc[0]
    assert row["mean_degree_syn"] > low["mean_degree_syn"]


# -------------------------------------------------------------------------- no SNAP names in code
SNAP_TOKENS = ("school", "birthday", "hometown", "employer", "gender", "locale", "languages", "location")


def test_no_snap_attribute_names_in_library_code():
    """No SNAP/Facebook attribute name appears anywhere in ``src/graph_aim`` (code, comments or
    docstrings): dataset knowledge must arrive through the schema and the policy, never the code."""
    src = Path(__file__).resolve().parents[2] / "src" / "graph_aim"
    assert src.is_dir()
    pattern = re.compile(r"\b(" + "|".join(SNAP_TOKENS) + r")\b")  # word boundary, case sensitive
    hits = []
    for p in sorted(src.rglob("*.py")):
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{p.relative_to(src)}:{lineno}: {line.strip()}")
    assert hits == [], "SNAP-specific attribute names leaked into library code:\n" + "\n".join(hits)
