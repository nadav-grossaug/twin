"""The generator across user counts: it must not crash at the small end of
its range, must produce exactly the requested population, must keep the
planted structures, and must meet the res-count mix tolerance at a
mid-size population."""
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]


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


@pytest.mark.parametrize("n", [25, 100, 400])
def test_small_populations_do_not_crash(gen, tmp_path, n):
    run, _ = gen
    r = run(n, tmp_path / f"o{n}")
    assert r.returncode == 0, (n, r.stderr[-300:])
    plan = json.loads((tmp_path / f"o{n}" / "plan.json").read_text())
    assert len(plan["users"]) == n
    assert plan["planted"]["return"]["user"] == 1
    assert plan["planted"]["tour"]["user"] == 2
    assert plan["planted"]["longstay"]["user"] == 3


def test_csv_rows_match_plan(gen, tmp_path):
    run, _ = gen
    n = 200
    r = run(n, tmp_path / "o")
    assert r.returncode == 0, r.stderr[-300:]
    plan = json.loads((tmp_path / "o" / "plan.json").read_text())
    rows = list(csv.DictReader(open(tmp_path / "o" / "booking_synth.csv")))
    assert sum(u["n_res"] for u in plan["users"]) == len(rows)
    got = Counter(r["user_id"] for r in rows)
    for u in plan["users"]:
        assert got[str(u["user"])] == u["n_res"]


def test_res_count_mix_within_tolerance_at_mid_population(gen, tmp_path):
    run, spec = gen
    n = 1000
    r = run(n, tmp_path / "o")
    assert r.returncode == 0, r.stderr[-300:]
    gs = json.loads(spec.read_text())
    rows = list(csv.DictReader(open(tmp_path / "o" / "booking_synth.csv")))
    got = Counter()
    for u in {r["user_id"] for r in rows}:
        n_res = sum(1 for r in rows if r["user_id"] == u)

        def rc_bin(k):
            if k <= 3: return "rc_1_3"
            if k == 4: return "rc_4"
            if k == 5: return "rc_5"
            if k == 6: return "rc_6"
            if k <= 8: return "rc_7_8"
            if k <= 12: return "rc_9_12"
            return "rc_13p"
        got[rc_bin(n_res)] += 1
    want = {c["name"]: c["w"] for c in gs["res_count"]["cells"]}
    tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
    tv = 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                   for c in RC_BINS)
    assert tv < 0.35, f"res_count TV {tv:.3f} at n={n}"


def test_repeated_runs_are_not_identical(gen, tmp_path):
    """Draws are urandom-backed (consistent with non-reproducible DP noise):
    two generator runs must differ in at least the city sequences."""
    run, _ = gen
    r1 = run(300, tmp_path / "r1")
    r2 = run(300, tmp_path / "r2")
    assert r1.returncode == 0 and r2.returncode == 0
    b1 = (tmp_path / "r1" / "booking_synth.csv").read_bytes()
    b2 = (tmp_path / "r2" / "booking_synth.csv").read_bytes()
    assert b1 != b2, "two generator runs produced identical output"
