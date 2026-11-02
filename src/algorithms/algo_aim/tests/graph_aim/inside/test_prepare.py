from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graph_aim.transfer.types import AttributedGraph
from prepare import (
    DATA_DIR,
    N_NODES,
    POLICY_PATH,
    build_attributed_graph,
    check_policy,
    compile_attributes,
    derive_policy,
    homophily_groups,
    load_attributed_graph,
    merge_egos,
    normalize_edges,
    parse_ego,
    save_attributed_graph,
    validate_attributed_graph,
)

HAVE_DATA = (DATA_DIR / "facebook_combined.txt.gz").exists() and (DATA_DIR / "facebook").exists()


# ---- synthetic fixture (no real data) ----

def _two_egos():
    # ego 0 sees nodes 1,2; ego 3 sees nodes 1,2 too (overlap); column order differs between egos
    e0 = parse_ego(
        ["1 1 0 0", "2 0 1 0"],
        ["0 gender;anonymized feature 7", "1 gender;anonymized feature 8", "2 school;id;anonymized feature 1"],
        "0 0 1",
        0,
    )
    e3 = parse_ego(
        ["1 0 1", "2 1 0"],
        ["0 school;id;anonymized feature 1", "1 gender;anonymized feature 8"],
        "1 0",
        3,
    )
    return e0, e3


def test_merge_maps_names_and_ors_overlaps():
    names, x = merge_egos(list(_two_egos()), n=4)
    assert names == sorted(names) and len(names) == 3
    j = {nm: i for i, nm in enumerate(names)}
    g7, g8, s1 = (j["gender;anonymized feature 7"], j["gender;anonymized feature 8"],
                  j["school;id;anonymized feature 1"])
    assert x[1, g7] and x[1, g8] and not x[1, s1]  # node 1: g7 from ego0, g8 from ego3 (OR)
    assert x[2, g8] and x[2, s1] and not x[2, g7]  # node 2: g8 from ego0, s1 from ego3
    assert x[0, s1] and not x[0, g7]  # ego 0 own features from egofeat
    assert x[3, s1] and not x[3, g8]  # ego 3 own features from egofeat


def test_parse_ego_rejects_width_mismatch():
    with pytest.raises(ValueError):
        parse_ego(["1 1 0"], ["0 a;anonymized feature 1"], "1", 0)


def test_compile_attributes_codes_and_policy():
    names, x = merge_egos(list(_two_egos()), n=4)
    policy = derive_policy(
        names, x, specs=[("sex", "gender", 1), ("school", "school;id", 1)], homophily_attrs=["sex"]
    )
    attrs, domains = compile_attributes(names, x, policy)
    assert domains["sex"][-2:] == ["other", "none"] and len(domains["sex"]) == 3
    # gender 8 is more frequent (nodes 2 and 1 via ego 3 and ego 0? count) -> single explicit value
    top = policy["attributes"][0]["values"][0]
    assert top.startswith("gender;")
    top_idx = [i for i, nm in enumerate(names) if nm == top][0]
    codes = attrs["sex"].to_numpy()
    for node in range(4):
        has_top = x[node, top_idx]
        has_any = x[node, [i for i, nm in enumerate(names) if nm.startswith("gender;")]].any()
        assert codes[node] == (0 if has_top else 1 if has_any else 2)
    assert (attrs["school"].to_numpy() == np.where(x[:, [i for i, nm in enumerate(names) if nm.startswith("school")][0]], 0, 2)).all()
    assert homophily_groups(policy) == 3


def test_policy_eval_keys_are_optional():
    names, x = merge_egos(list(_two_egos()), n=4)
    policy = derive_policy(names, x, specs=[("sex", "gender", 1)], homophily_attrs=["sex"])
    assert policy["attributes"][0]["missing_label"] == "none" and policy["attributes"][0]["ordered"] is False
    old = {**policy, "attributes": [{k: v for k, v in a.items() if k not in ("missing_label", "ordered")} for a in policy["attributes"]]}
    check_policy(old)  # policies written before the keys existed still load and compile
    a_old, d_old = compile_attributes(names, x, old)
    a_new, d_new = compile_attributes(names, x, policy)
    assert a_old.equals(a_new) and d_old == d_new


def test_policy_rejects_identifier_and_oversized():
    names, x = merge_egos(list(_two_egos()), n=4)
    with pytest.raises(ValueError):
        check_policy(derive_policy(names, x, specs=[("n", "first_name", 1)], homophily_attrs=[]))
    with pytest.raises(ValueError):
        check_policy(derive_policy(names, x, specs=[("g", "gender", 7)], homophily_attrs=[]))
    with pytest.raises(ValueError):
        check_policy(derive_policy(names, x, specs=[("g", "gender", 1)], homophily_attrs=["zzz"]))


def test_normalize_edges():
    e = normalize_edges(np.array([[2, 1], [1, 2], [3, 3], [0, 4]]))
    assert e.tolist() == [[0, 4], [1, 2]]


def _small_graph() -> AttributedGraph:
    return AttributedGraph(
        n=3,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        attrs=pd.DataFrame({"a": [0, 1, 2]}, index=pd.RangeIndex(3)),
        domains={"a": ["x", "other", "none"]},
    )


def test_validate_catches_violations():
    validate_attributed_graph(_small_graph(), 3)
    for mutate in (
        lambda g: setattr(g, "edges", np.array([[1, 0]], dtype=np.int64)),
        lambda g: setattr(g, "edges", np.array([[0, 1], [0, 1]], dtype=np.int64)),
        lambda g: setattr(g, "edges", np.array([[1, 1]], dtype=np.int64)),
        lambda g: setattr(g, "edges", np.array([[0, 5]], dtype=np.int64)),
        lambda g: g.attrs.__setitem__("a", [0, -1, 2]),
        lambda g: g.attrs.__setitem__("a", [0, 1, 3]),
        lambda g: g.domains.__setitem__("a", ["x", "y", "none"]),
    ):
        g = _small_graph()
        mutate(g)
        with pytest.raises(ValueError):
            validate_attributed_graph(g)
    with pytest.raises(ValueError):
        validate_attributed_graph(_small_graph(), 4)


def test_save_load_roundtrip(tmp_path: Path):
    g = _small_graph()
    p = tmp_path / "g.npz"
    save_attributed_graph(g, p)
    h = load_attributed_graph(p)
    assert h.n == g.n and h.domains == g.domains
    assert np.array_equal(h.edges, g.edges)
    pd.testing.assert_frame_equal(h.attrs, g.attrs)
    validate_attributed_graph(h)


# ---- real data ----

@pytest.fixture(scope="module")
def real() -> AttributedGraph:
    if not HAVE_DATA:
        pytest.skip("SNAP data not present")
    return build_attributed_graph(DATA_DIR, POLICY_PATH)


@pytest.mark.skipif(not HAVE_DATA, reason="SNAP data not present")
def test_real_graph_invariants(real: AttributedGraph):
    validate_attributed_graph(real, N_NODES)
    assert real.n == 4039 and len(real.edges) == 88234
    assert 1 <= real.attrs.shape[1] <= 8
    assert all(len(d) <= 8 for d in real.domains.values())
    assert (real.attrs.to_numpy() >= 0).all()


@pytest.mark.skipif(not HAVE_DATA, reason="SNAP data not present")
def test_real_policy_and_determinism(real: AttributedGraph):
    policy = json.loads(POLICY_PATH.read_text())
    check_policy(policy)
    assert set(policy["dropped_fields"]) == {"first_name", "middle_name", "last_name", "name"}
    assert 1 <= len(policy["homophily_attrs"]) <= 2 and homophily_groups(policy) <= 6
    assert "outside the DP guarantee" in policy["note"]
    assert list(real.attrs.columns) == [a["name"] for a in policy["attributes"]]
    again = build_attributed_graph(DATA_DIR, POLICY_PATH)
    assert np.array_equal(again.edges, real.edges) and again.attrs.equals(real.attrs)


@pytest.mark.skipif(not HAVE_DATA, reason="SNAP data not present")
def test_real_processed_file_matches_build(real: AttributedGraph):
    p = DATA_DIR / "processed" / "attributed_graph.npz"
    if not p.exists():
        pytest.skip("processed file not generated")
    saved = load_attributed_graph(p)
    assert saved.domains == real.domains and saved.attrs.equals(real.attrs)
    assert np.array_equal(saved.edges, real.edges)
