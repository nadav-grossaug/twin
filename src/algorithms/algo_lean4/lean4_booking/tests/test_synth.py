"""Synthetic booking data: meta/provenance, planted-pattern recall, per-column
structural and distributional checks on the generated CSV."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timedelta

import pytest

from spec import load_spec

COLS = ["user_id", "checkin", "checkout", "city_id", "device_class",
        "affiliate_id", "booker_country", "hotel_country", "utrip_id"]
RC_BINS = {"rc_1_3": (1, 3), "rc_4": (4, 4), "rc_5": (5, 5), "rc_6": (6, 6),
           "rc_7_8": (7, 8), "rc_9_12": (9, 12), "rc_13p": (13, 10 ** 6)}
CT_CELLS = ["Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib",
            "Elbonia", "Borginia", "Kangan", "Bozatta", "Alvonia",
            "Holy Britannian Empire", "Axphain", "Santa Prisca", "Patusan",
            "Rolisica", "Kasnia", "ct_other"]


@pytest.fixture(scope="module")
def synth(root):
    s = root / "synth"
    if not (s / "booking_synth.csv").exists():
        pytest.skip("run `make synth` first")
    rows = list(csv.DictReader(open(s / "booking_synth.csv")))
    plan = json.loads((s / "plan.json").read_text())
    plan.setdefault("planted", {})
    meta = json.loads((s / "meta.json").read_text())
    return s, rows, plan, meta


def _d(s):
    return date.fromisoformat(s)


def test_meta_declares_gaps_and_provenance(synth):
    _, _, _, meta = synth
    gaps = meta["assumptions"]["fields_not_modelled"]
    assert not any("multi-trip" in g for g in gaps)  # 1.3.0: released
    assert any("within-trip affiliate order" in g for g in gaps)
    assert any("top-k" in g for g in gaps)
    assert any("day-of-week" in g for g in gaps)
    assert any("affiliate" in g for g in gaps)
    assert meta["genspec_sha256"] and meta["dp_release_sha256"]


def test_schema_and_structure(synth):
    """The structural invariants the Lean generator proves by construction,
    re-checked empirically on the rendered CSV."""
    _, rows, _, _ = synth
    assert list(rows[0].keys()) == COLS
    spec = load_spec()
    w0 = date(1970, 1, 1) + timedelta(days=int(spec["window_start_day"]))
    w1 = date(1970, 1, 1) + timedelta(days=int(spec["window_end_day"]))
    by_user = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)
    for u, rs in by_user.items():
        # 1.3.0 multi-trip: every trip is "<user>_<k>" with the k's
        # forming a gapless 1..n sequence
        utrips = sorted({r["utrip_id"] for r in rs})
        want = [f"{u}_{k}" for k in range(1, len(utrips) + 1)]
        assert utrips == want, u
        rs = sorted(rs, key=lambda r: r["checkin"])
        checkins = [_d(r["checkin"]) for r in rs]
        assert checkins == sorted(checkins) and len(set(checkins)) == len(checkins), (
            f"user {u}: checkins not strictly increasing")
        for r in rs:
            ci, co = _d(r["checkin"]), _d(r["checkout"])
            assert co > ci, (u, r)              # checkout strictly after checkin
            nights = (co - ci).days
            assert 1 <= nights <= 30, (u, nights)
            assert r["device_class"] in ("desktop", "mobile", "tablet")
            assert r["booker_country"] and r["hotel_country"]
        assert w0 <= checkins[0] and checkins[-1] < w1, (
            f"user {u} outside public window")
        # NO OVERLAPPING PRESENCE: a stay never extends past the next checkin
        for a, b in zip(rs, rs[1:]):
            assert _d(a["checkout"]) <= _d(b["checkin"]), (
                f"user {u}: overlapping presence")
        assert _d(rs[-1]["checkout"]) <= w1, (
            f"user {u}: last stay extends past the window")


def test_res_counts_match_plan_and_bins(synth):
    _, rows, plan, meta = synth
    assert len(plan["users"]) == meta["n_users"]
    got = Counter(r["user_id"] for r in rows)
    for u in plan["users"]:
        assert got[str(u["user"])] == u["n_res"], u["user"]
        # every planned count sits inside SOME public bin (1..rc_13p cap)
        assert any(lo <= u["n_res"] <= hi for lo, hi in RC_BINS.values()), u


def test_planted_return_in_plan(synth):
    _, rows, plan, _ = synth
    c = plan["planted"]["return"]
    user_rows = [r for r in rows if r["user_id"] == str(c["user"])]
    cities = [r["city_id"] for r in user_rows]
    assert len(set(cities)) < len(cities), (
        f"planted return user {c['user']} has no repeat city")


def test_planted_tour_in_plan(synth):
    _, rows, plan, _ = synth
    t = plan["planted"]["tour"]
    user_rows = sorted((r for r in rows if r["user_id"] == str(t["user"])),
                       key=lambda r: r["checkin"])
    assert len(user_rows) >= 4
    countries = {r["hotel_country"] for r in user_rows}
    assert len(countries) >= 3
    days = [_d(r["checkin"]) for r in user_rows]
    for a, b in zip(days, days[1:]):
        assert (b - a).days <= 7, "planted tour legs more than 7d apart"


def test_planted_longstay_in_plan(synth):
    _, rows, plan, _ = synth
    ls = plan["planted"]["longstay"]
    user_rows = [r for r in rows if r["user_id"] == str(ls["user"])]
    nights = [(_d(r["checkout"]) - _d(r["checkin"])).days for r in user_rows]
    assert max(nights) >= 7, f"planted longstay user {ls['user']} max {max(nights)}"


def test_nights_mix_follows_genspec(synth, root):
    _, rows, _, meta = synth
    gs_p = root / "lean" / "out" / "genspec.json"
    gs = json.loads(gs_p.read_text())
    if meta["genspec_sha256"] != hashlib.sha256(gs_p.read_bytes()).hexdigest():
        pytest.skip("synth predates the current genspec — rerun `make synth`")

    def night_cell(n):
        if n == 1: return "n_1"
        if n == 2: return "n_2"
        if n == 3: return "n_3"
        if n <= 6: return "n_4_6"
        return "n_7p"

    got = Counter(night_cell((_d(r["checkout"]) - _d(r["checkin"])).days)
                  for r in rows)
    want = {c["name"]: c["w"] for c in gs["nights"]["cells"]}
    tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
    tv = 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                   for c in ["n_1", "n_2", "n_3", "n_4_6", "n_7p"])
    assert tv < 0.25, f"nights mix TV {tv:.3f}"


def test_country_mix_tracks_genspec(synth, root):
    _, rows, _, meta = synth
    gs_p = root / "lean" / "out" / "genspec.json"
    gs = json.loads(gs_p.read_text())
    if meta["genspec_sha256"] != hashlib.sha256(gs_p.read_bytes()).hexdigest():
        pytest.skip("synth predates the current genspec — rerun `make synth`")

    def ct_of(c):
        return c if c in CT_CELLS else "ct_other"

    got = Counter(ct_of(r["hotel_country"]) for r in rows)
    want = {c["name"]: c["w"] for c in gs["country_mix"]["cells"]}
    tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
    tv = 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                   for c in CT_CELLS)
    # the generator blends transitions, returns and released cities on top
    # of the marginal mixes, so tolerate drift
    assert tv < 0.35, f"country mix TV {tv:.3f}"


def test_validation_passed(synth):
    s, _, plan, _ = synth
    p = s / "validation.json"
    if not p.exists():
        pytest.skip("run `make synth` (validate step) first")
    v = json.loads(p.read_text())
    assert not v["hard_failures"], v["hard_failures"]
    if plan["planted"].get("return"):
        assert v["notes"]["planted_return_recall"] == 1.0
    if plan["planted"].get("tour"):
        assert v["notes"]["planted_tour_recall"] == 1.0
    if plan["planted"].get("longstay"):
        assert v["notes"]["planted_longstay_recall"] == 1.0


def test_certificate_kernel_checked(synth):
    s, _, _, _ = synth
    p = s / "certificate.json"
    if not p.exists():
        pytest.skip("run `make certify` first")
    cert = json.loads(p.read_text())
    assert cert["proved"] is True
    assert cert["kernel_only"] is True
    for thm in ("planted_is_return", "planted_is_tour",
                "planted_is_longstay", "planned_counts_in_bins"):
        assert thm in cert["theorems"], thm
