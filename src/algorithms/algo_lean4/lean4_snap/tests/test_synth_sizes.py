"""The generator across node counts: it must not crash at the small end of
its documented range, must warn outside the validated fidelity window, and
must meet the reference degree tolerance inside it."""
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter

import pytest

NS = "{http://graphml.graphdrawing.org/xmlns}"
EDGES = [(1, "deg_1"), (2, "deg_2"), (3, "deg_3_4"), (5, "deg_5_8"),
         (9, "deg_9_16"), (17, "deg_17_32"), (33, "deg_33_64"), (65, "deg_65p")]


@pytest.fixture(scope="module")
def gen(root):
    binp = root / "lean" / ".lake" / "build" / "bin" / "generate"
    spec = root / "lean" / "out" / "genspec.json"
    if not binp.exists() or not spec.exists():
        pytest.skip("run `make synth` first (generator or genspec missing)")

    def run(n, out):
        return subprocess.run([str(binp), str(spec), str(n), str(out)],
                              cwd=str(root / "lean"), capture_output=True, text=True)
    return run, spec


@pytest.mark.parametrize("n", [200, 250, 300])
def test_small_graphs_plant_a_verified_broker(gen, tmp_path, n):
    """Regression: at n <~ 350 the strict zero-common-neighbor broker search
    ran out of candidates (generate exited 1: 'broker planting failed'). The
    greedy fewest-common-neighbors fallback must now always succeed."""
    run, _ = gen
    for i in range(4):
        r = run(n, tmp_path / f"o{i}")
        assert r.returncode == 0, (n, i, r.stderr[-200:])
        plan = json.loads((tmp_path / f"o{i}" / "plan.json").read_text())
        assert plan["planted"]["broker"]["degree"] >= 17


def test_warns_below_the_validated_window(gen, tmp_path):
    run, _ = gen
    r = run(200, tmp_path / "o")
    assert "below the validated fidelity range" in r.stderr


def test_warns_above_the_validated_window(gen, tmp_path):
    run, _ = gen
    r = run(10500, tmp_path / "o")
    assert r.returncode == 0
    assert "above the validated fidelity range" in r.stderr


def test_no_warning_inside_the_window(gen, tmp_path):
    run, _ = gen
    r = run(1000, tmp_path / "o")
    assert r.returncode == 0 and "validated fidelity range" not in r.stderr


def _degree_tv(graphml, spec):
    deg = Counter()
    root = ET.parse(graphml).getroot()
    nodes = [nd.get("id") for nd in root.iter(f"{NS}node")]
    for e in root.iter(f"{NS}edge"):
        deg[e.get("source")] += 1
        deg[e.get("target")] += 1

    def cls(d):
        out = "deg_1"
        for lo, name in EDGES:
            if d >= lo:
                out = name
        return out
    got = Counter(cls(deg[v]) for v in nodes)
    want = {c["name"]: c["w"] for c in json.loads(spec.read_text())["degree"]["cells"]}
    tw, tg = sum(want.values()), sum(got.values())
    return 0.5 * sum(abs(want[k] / tw - got.get(k, 0) / tg) for k in want)


@pytest.mark.parametrize("n", [500, 1000])
def test_degree_mix_within_tolerance_inside_the_window(gen, tmp_path, n):
    """Measured 18/18 runs at 500..1000 stayed <= 0.205 (tolerance 0.25)."""
    run, spec = gen
    for i in range(3):
        assert run(n, tmp_path / f"o{i}").returncode == 0
        tv = _degree_tv(tmp_path / f"o{i}" / "synth_graph.graphml", spec)
        assert tv <= 0.25, (n, i, tv)
