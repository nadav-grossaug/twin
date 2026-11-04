"""Release gate: default deny, exact-fraction ledger, policy-filtered export."""
import json
from fractions import Fraction

import pytest

from release import epsilon_ledger, is_allowed, load_policy


@pytest.fixture(scope="module")
def policy():
    return load_policy()


def test_ledger_within_budget(policy):
    ledger = epsilon_ledger(policy)
    assert Fraction(ledger["epsilon_total"]) <= Fraction(ledger["epsilon_max"])


def test_ledger_matches_the_contract(policy):
    # wiki/booking-contract.md: the 1.4.0 ledger sums to exactly
    # 1440/100 = 72/5 = 14.4 (the owner raised the cap to 16/1; 1.4.0 adds
    # countries_per_trip, device_plan, domestic_start)
    ledger = epsilon_ledger(policy)
    assert Fraction(ledger["epsilon_total"]) == Fraction("1440/100")
    assert Fraction(ledger["epsilon_total"]) <= Fraction(ledger["epsilon_max"])
    assert Fraction(ledger["epsilon_max"]) == Fraction("16/1")
    assert "weak" in ledger["note"]


def test_over_budget_refused(policy):
    bad = json.loads(json.dumps(policy))
    bad["dp_releases"]["country_transition_histogram"]["epsilon_per_cell"] = "2/1"
    with pytest.raises(PermissionError):
        epsilon_ledger(bad)


@pytest.mark.parametrize("relation", [
    "stg_reservations", "feat_reservations", "feat_transitions",
    "feat_trips", "feat_users", "ml_trip_scores", "mart_user_stats",
    "mart_trip_stats", "mart_motif_returns", "mart_motif_corridors",
    "mart_motif_tours", "mart_motif_longstays", "mart_assoc_cells",
    "mart_seq_transitions",
])
def test_default_deny(policy, relation):
    assert not is_allowed(relation, policy)
    assert relation in policy["denied"], f"{relation} missing from the audit list"


def test_allowed_artifacts(policy):
    assert is_allowed("mart_topk_cities", policy)
    assert is_allowed("mart_topk_city_transitions", policy)
    assert is_allowed("mart_gap_moments", policy)


def test_model_weights_denied(policy):
    assert policy["model_weights"] == "deny_unless_reviewed"


def test_dp_release_grids_match_lean_output(root, policy):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make train`")
    rel = json.loads(p.read_text())
    assert len(rel["cells"]) == policy["dp_releases"]["res_count_histogram"]["n_cells"]
    by_name = {h["name"]: h for h in rel["aux"]}
    for pol_name, aux_name in [
        ("country_histogram", "country"),
        ("country_transition_histogram", "country_transition"),
        ("nights_histogram", "nights"),
        ("gap_histogram", "gap"),
        ("month_window_histogram", "month_window"),
        ("repeat_struct_histogram", "repeat_struct"),
        ("trips_per_user_histogram", "trips_per_user"),
        ("device_persist_histogram", "device_persist"),
        ("affiliate_persist_histogram", "affiliate_persist"),
        ("device_histogram", "device"),
        ("affiliate_histogram", "affiliate"),
        ("booker_histogram", "booker"),
        ("return_rate_histogram", "return_rate"),
    ]:
        spec = policy["dp_releases"][pol_name]
        aux = by_name[aux_name]
        assert len(aux["cells"]) == spec["n_cells"], pol_name
        assert Fraction(aux["eps1"], aux["eps2"]) == Fraction(
            spec["epsilon_per_cell"]), pol_name


def test_released_files_have_no_user_tokens(root):
    """Released artifacts must clear the suppression thresholds; user ids,
    trip ids and per-user rows never appear in any released JSON."""
    rel_dirs = sorted((root / "release").glob("*/")) if (root / "release").exists() else []
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    latest = rel_dirs[-1]
    for f in latest.glob("*.json"):
        txt = f.read_text()
        assert "userid" not in txt, f.name
        assert "user_id" not in txt, f.name
        assert "utrip" not in txt, f.name
    for f in latest.glob("topk_*.csv"):
        # exact column names, not substrings: n_users (a suppression COUNT)
        # is fine; an id column named user/userid/utrip is not
        cols = f.open().readline().strip().split(",")
        forbidden = {"user", "userid", "user_id", "utrip", "utrip_id"}
        assert not (set(cols) & forbidden), (f.name, cols)


# ---------------------------------------------------------------------------
# lifetime epsilon ledger (P2): every export of the same data composes
# ---------------------------------------------------------------------------

def test_lifetime_ledger_seeded_and_wellformed(root):
    """If a ledger exists, every entry parses as an exact Fraction and the
    recorded lifetime totals are consistent with the entries."""
    from release import lifetime_entries
    entries = lifetime_entries()
    if not entries:
        pytest.skip("no exports yet — ledger not seeded")
    seen = []
    for e in entries:
        assert e["data_sha256"] and e["export_id"]
        assert Fraction(e["epsilon_spent"]) > 0
        seen.append(Fraction(e["epsilon_spent"]))
        # lifetime_total must equal the running sum for its data sha
        same = [Fraction(x["epsilon_spent"]) for x in entries
                if x["data_sha256"] == e["data_sha256"]
                and entries.index(x) <= entries.index(e)]
        assert Fraction(e["lifetime_total"]) == sum(same)


def test_lifetime_budget_refuses_over_cap(tmp_path):
    """Cumulative spend over the policy's lifetime cap refuses the export."""
    import json as _json
    from release import append_lifetime_entry, check_lifetime_budget
    policy = {"dp_lifetime_epsilon_max": "21/1"}   # two exports of 1008/100 fit
    sha = "d4ta"
    check_lifetime_budget(policy, sha, Fraction("1008/100"))  # 10.08 ok
    led = tmp_path / "epsilon_ledger.jsonl"
    append_lifetime_entry({"export_id": "x1", "data_sha256": sha,
                           "epsilon_spent": "1008/100"}, path=led)
    # second export of the same data: 20.16 <= 21 ok — and it APPENDS on
    # success (export() appends; the check alone never writes)
    check_lifetime_budget(policy, sha, Fraction("1008/100"), path=led)
    append_lifetime_entry({"export_id": "x2", "data_sha256": sha,
                           "epsilon_spent": "1008/100"}, path=led)
    # third: 30.24 > 21 -> refused
    with pytest.raises(PermissionError, match="LIFETIME"):
        check_lifetime_budget(policy, sha, Fraction("1008/100"), path=led)
    # a DIFFERENT dataset starts fresh
    check_lifetime_budget(policy, "other", Fraction("1008/100"), path=led)


def test_lifetime_budget_zero_cap_means_no_cap(tmp_path):
    """A policy without dp_lifetime_epsilon_max does not gate (cap 0)."""
    from release import check_lifetime_budget
    out = check_lifetime_budget({}, "sha", Fraction("1008/100"),
                                path=tmp_path / "none.jsonl")
    assert Fraction(out["after_this_export"]) == Fraction("1008/100")
    assert out["cap"] is None
