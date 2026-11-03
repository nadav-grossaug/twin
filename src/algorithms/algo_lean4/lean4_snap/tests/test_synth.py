"""Synthetic graph: meta/provenance, planted-pattern recall, structural and
distributional checks on the generated GraphML."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter

import pytest

from spec import load_spec

NS = "{http://graphml.graphdrawing.org/xmlns}"
DEG_BINS = {"deg_1": (1, 1), "deg_2": (2, 2), "deg_3_4": (3, 4),
            "deg_5_8": (5, 8), "deg_9_16": (9, 16), "deg_17_32": (17, 32),
            "deg_33_64": (33, 64), "deg_65p": (65, 10 ** 6)}


@pytest.fixture(scope="module")
def synth(root):
    s = root / "synth"
    if not (s / "synth_graph.graphml").exists():
        pytest.skip("run `make synth` first")
    tree = ET.parse(s / "synth_graph.graphml")
    plan = json.loads((s / "plan.json").read_text())
    meta = json.loads((s / "meta.json").read_text())
    return s, tree, plan, meta


def _edges(tree):
    return [(int(e.get("source")), int(e.get("target")))
            for e in tree.getroot().iter(f"{NS}edge")]


def _nodes(tree):
    return list(tree.getroot().iter(f"{NS}node"))


def test_meta_declares_gaps_and_provenance(synth):
    _, _, _, meta = synth
    gaps = meta["assumptions"]["fields_not_modelled"]
    assert any("assortativity" in g for g in gaps)
    assert any("circle" in g for g in gaps)
    assert meta["genspec_sha256"] and meta["dp_release_sha256"]


def test_edges_canonical_and_unique(synth):
    """The structural invariants the Lean generator proves by construction
    (mkEdge_ok, edgesCanonical_sound, edges_nodup), re-checked empirically."""
    _, tree, plan, _ = synth
    edges = _edges(tree)
    assert len(edges) == plan["n_edges"]
    n = plan["n_nodes"]
    seen = set()
    for a, b in edges:
        assert 0 <= a < b < n, f"non-canonical edge ({a}, {b})"
        assert (a, b) not in seen, f"duplicate edge ({a}, {b})"
        seen.add((a, b))


def test_node_count_and_ego_marking(synth):
    _, tree, plan, _ = synth
    nodes = _nodes(tree)
    assert len(nodes) == plan["n_nodes"]
    ego_ids = set(plan["ego_ids"])
    marked = {int(nd.get("id")) for nd in nodes
              for d in nd.iter(f"{NS}data")
              if d.get("key") == "d0_is_ego" and d.text == "true"}
    assert marked == ego_ids


def test_planted_hub(synth):
    _, tree, plan, _ = synth
    deg = Counter()
    for a, b in _edges(tree):
        deg[a] += 1
        deg[b] += 1
    hub = plan["planted"]["hub"]
    assert deg[int(hub["node"])] >= int(load_spec()["hub_degree_min"])
    assert deg[int(hub["node"])] == int(hub["degree"])


def test_planted_broker_facts(synth):
    """plan.json carries the broker's (degree, triangles); the IsBroker
    inequality must hold on those numbers (the generator verified it
    decidably; the certificate re-proves it in the kernel)."""
    _, _, plan, _ = synth
    br = plan["planted"]["broker"]
    s = load_spec()
    d, t = int(br["degree"]), int(br["triangles"])
    assert d >= int(s["broker_degree_min"])
    assert int(s["broker_cc_max_den"]) * 2 * t < \
        int(s["broker_cc_max_num"]) * d * (d - 1)


def test_planted_butterfly(synth):
    _, tree, plan, _ = synth
    bf = plan["planted"]["butterfly"]
    assert int(bf["circles"]) >= int(load_spec()["butterfly_circles_min"])
    node = str(bf["node"])
    for nd in _nodes(tree):
        if nd.get("id") == node:
            circ = [d.text for d in nd.iter(f"{NS}data")
                    if d.get("key", "").endswith("_circles")]
            assert circ and len(circ[0].split("|")) == int(bf["circles"])
            return
    pytest.fail(f"butterfly node {node} not found")


def test_feature_keys_match_public_shape(synth):
    _, tree, _, _ = synth
    pat = re.compile(r"^ego_\d+__f\d+_\w+_anonymized_feature_\d+$")
    keys = [k.get("attr.name") for k in tree.getroot().iter(f"{NS}key")]
    feat_keys = [k for k in keys if k and k.startswith("ego_")
                 and "anonymized_feature" in k]
    assert feat_keys, "no feature keys rendered"
    for k in feat_keys:
        assert pat.match(k), k


def test_degree_mix_tracks_genspec(synth, root):
    _, tree, _, meta = synth
    gs_p = root / "lean" / "out" / "genspec.json"
    gs = json.loads(gs_p.read_text())
    if meta["genspec_sha256"] != hashlib.sha256(gs_p.read_bytes()).hexdigest():
        pytest.skip("synth predates the current genspec — rerun `make synth`")
    deg = Counter()
    for a, b in _edges(tree):
        deg[a] += 1
        deg[b] += 1

    def cls(d):
        for name, (lo, hi) in DEG_BINS.items():
            if lo <= d <= hi:
                return name
        return "deg_1"

    got = Counter(cls(d) for d in deg.values())
    want = {c["name"]: c["w"] for c in gs["degree"]["cells"]}
    tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
    tv = 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                   for c in DEG_BINS)
    # the closure pass and planting shift degrees upward; tolerate drift
    assert tv < 0.35, f"degree mix TV {tv:.3f}"


def test_validation_passed(synth):
    s, _, _, _ = synth
    p = s / "validation.json"
    if not p.exists():
        pytest.skip("run `make synth` (validate step) first")
    v = json.loads(p.read_text())
    assert not v["hard_failures"], v["hard_failures"]
    for motif in ("hub", "broker", "butterfly"):
        assert v["notes"][f"planted_{motif}_recall"] == 1.0, motif


def test_certificate_kernel_checked(synth):
    s, _, _, _ = synth
    p = s / "certificate.json"
    if not p.exists():
        pytest.skip("run `make certify` first")
    cert = json.loads(p.read_text())
    assert cert["proved"] is True
    assert cert["kernel_only"] is True
