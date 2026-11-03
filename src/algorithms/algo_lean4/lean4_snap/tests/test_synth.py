"""Synthetic graph: meta/provenance, planted-pattern recall, structural and
distributional checks on the generated GraphML."""
import hashlib
import json
import re
import warnings
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
    # soft tolerances (closure is the known marginal one) must not be hidden:
    # a single sample of a stochastic generator would make a hard assert flaky,
    # so they surface as pytest warnings. Affiliation checks are asserted below.
    other = [f for f in v["soft_failures"] if not f.startswith("affil")]
    if other:
        warnings.warn(f"synth validation soft failures (not asserted): {other}")


def test_replicate_verdicts(synth):
    """Closure and affiliation density are properties of the GENERATOR, judged
    over replicate runs (`make synth` runs `synth.py replicates`): closure needs
    enough runs inside the band with the mean clear of its lower edge; the mean
    |synthetic - released| group density must stay under its limit."""
    s, _, _, _ = synth
    p = s / "replicates.json"
    if not p.exists():
        pytest.skip("run `make synth` (replicates step) first")
    r = json.loads(p.read_text())
    assert r["n_runs"] >= 6, r["n_runs"]
    assert r.get("closure_passed", r["passed"]), {k: r[k] for k in (
        "mean", "min", "n_in_band", "n_runs", "margin_over_lower_edge", "rule")}
    if "affiliation_density" in r:
        assert r["affiliation_density"]["passed"], r["affiliation_density"]
    assert r["passed"]


def test_affiliation_checks_pass_and_are_evidenced(synth):
    """The validator's affiliation verdicts must be clean AND backed by
    per-cell evidence: group-count mix, coverage and per-bin density."""
    s, _, plan, _ = synth
    p = s / "validation.json"
    if not p.exists():
        pytest.skip("run `make synth` (validate step) first")
    v = json.loads(p.read_text())
    if "affil_cells" not in v["notes"]:
        pytest.skip("synth predates the affiliation validator or spec has no affiliation")
    # stable properties are asserted per run; per-cell DENSITY on one sample of a
    # random generator is noisy (a healthy run trips a cell ~1 time in 12), so it
    # surfaces as a warning and the authoritative density verdict is the
    # replicate one (test_replicate_verdicts).
    affil_soft = [f for f in v["soft_failures"] if f.startswith("affil")]
    density_flags = [f for f in affil_soft if f.startswith("affil density")]
    stable = [f for f in affil_soft if f not in density_flags]
    assert not stable, stable
    if density_flags:
        warnings.warn(f"single-run affiliation density flags (not asserted): {density_flags}")
    cats_with_tv = [k for k in v["notes"] if k.startswith("affil_group_count_tv_")]
    assert len(cats_with_tv) >= 5, cats_with_tv
    for k in cats_with_tv:
        tol = v["notes"][k.replace("_tv_", "_tol_")]
        assert v["notes"][k] <= tol, (k, v["notes"][k], tol)
    judged = [c for c in v["notes"]["affil_cells"].values()
              if c["n_groups"] >= 20 and c["released_density"] is not None]
    assert judged, "no affiliation cell had enough groups to be judged"


def test_affiliation_groups_are_cohesive(synth, root):
    """Independent of the validator: recompute key-group density straight
    from the rendered GraphML. Shared education/work keys must induce groups
    whose mean internal density is far above the global edge density."""
    _, tree, plan, meta = synth
    aff = plan.get("affiliation")
    gs = json.loads((root / "lean" / "out" / "genspec.json").read_text())
    if not aff or not aff.get("modelled") or "affiliation" not in gs:
        pytest.skip("affiliation not modelled by this genspec")
    assert aff["groups"] > 0 and aff["members"] > 0
    holders: dict[str, list[int]] = {}
    for nd in _nodes(tree):
        v = int(nd.get("id"))
        for d in nd.iter(f"{NS}data"):
            k = d.get("key", "")
            if any(k.endswith("_" + c) for c in gs["affiliation"]["categories"]):
                holders.setdefault(k, []).append(v)
    groups = {k: hs for k, hs in holders.items() if len(hs) >= 2}
    assert groups, "no shared education/work keys rendered"
    # the planted group count must match the plan (no phantom / lost groups)
    assert len(groups) == aff["groups"], (len(groups), aff["groups"])
    edges = set(_edges(tree))
    n = plan["n_nodes"]
    global_density = plan["n_edges"] / (n * (n - 1) / 2)
    dens = []
    for hs in groups.values():
        pairs = [(a, b) for i, a in enumerate(hs) for b in hs[i + 1:]]
        internal = sum(1 for a, b in pairs if (min(a, b), max(a, b)) in edges)
        dens.append(internal / len(pairs))
    mean_density = sum(dens) / len(dens)
    assert mean_density > 10 * global_density, (
        f"affiliation groups not cohesive: {mean_density:.4f} vs "
        f"global {global_density:.4f}")


def test_certificate_kernel_checked(synth):
    s, _, _, _ = synth
    p = s / "certificate.json"
    if not p.exists():
        pytest.skip("run `make certify` first")
    cert = json.loads(p.read_text())
    assert cert["proved"] is True
    assert cert["kernel_only"] is True
