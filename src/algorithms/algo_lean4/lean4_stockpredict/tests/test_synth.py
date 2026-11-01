"""Synthetic series: meta/provenance, planted-pattern recall, per-column
structural and distributional checks on the generated TSV."""
import csv
import json
import math
from collections import Counter
from datetime import date, timedelta

import pytest

CLASSES = ["crash", "down", "flat", "up", "surge"]


@pytest.fixture(scope="module")
def synth(root):
    s = root / "synth"
    if not (s / "stock_synth.tsv").exists():
        pytest.skip("run `make synth` first")
    rows = list(csv.DictReader(open(s / "stock_synth.tsv"), delimiter="\t"))
    plan = json.loads((s / "plan.json").read_text())
    meta = json.loads((s / "meta.json").read_text())
    return s, rows, plan, meta


def test_meta_declares_gaps_and_provenance(synth):
    _, _, _, meta = synth
    gaps = meta["assumptions"]["fields_not_modelled"]
    assert any("correlation" in g for g in gaps)
    assert any("volume" in g for g in gaps)
    assert meta["genspec_sha256"] and meta["dp_release_sha256"]


def test_schema_and_structure(synth):
    _, rows, plan, _ = synth
    assert set(rows[0].keys()) == {"stockname", "date-yyyy-mm-dd", "stockprice"}
    by_ticker: dict[str, list] = {}
    for r in rows:
        by_ticker.setdefault(r["stockname"], []).append(r)
    assert len(by_ticker) == plan["n_tickers"]
    for t, rs in by_ticker.items():
        prices = [float(r["stockprice"]) for r in rs]
        dates = [date.fromisoformat(r["date-yyyy-mm-dd"]) for r in rs]
        assert len(rs) == plan["n_days"]
        assert abs(prices[0] - 100.0) < 1e-9, f"{t} does not start at 100"
        assert all(p > 0 for p in prices)
        assert dates == sorted(dates)
        assert all(d.isoweekday() <= 5 for d in dates), f"{t} has weekend dates"
        # weekday calendar: consecutive gap is 1 day, or 3 over a weekend
        for a, b in zip(dates, dates[1:]):
            assert b - a in (timedelta(days=1), timedelta(days=3))


def test_planned_classes_match_taxonomy(synth):
    _, _, plan, _ = synth

    def class_of(bp):
        if bp <= -300:
            return "crash"
        if bp < -50:
            return "down"
        if bp <= 50:
            return "flat"
        if bp < 300:
            return "up"
        return "surge"

    for t in plan["tickers"]:
        for bp, cls in zip(t["rets_bp"], t["classes"]):
            assert class_of(bp) == cls


def test_prices_match_planned_log_returns(synth):
    _, rows, plan, _ = synth
    by_ticker: dict[str, list] = {}
    for r in rows:
        by_ticker.setdefault(r["stockname"], []).append(float(r["stockprice"]))
    for t in plan["tickers"]:
        prices = by_ticker[t["ticker"]]
        cum = 0
        for i, bp in enumerate(t["rets_bp"]):
            cum += bp
            want = 100.0 * math.exp(cum / 10000.0)
            assert abs(prices[i + 1] - want) < 1e-3, (t["ticker"], i)


def test_planted_pattern_in_plan(synth):
    _, _, plan, _ = synth
    planted = plan["planted"][0]
    t = next(x for x in plan["tickers"] if x["ticker"] == planted["ticker"])
    start = planted["start_idx"] - 1   # rets index 0 = day 1
    seg = t["rets_bp"][start:start + planted["run_days"]]
    assert all(bp < -50 for bp in seg), "planted run must be all-down"
    assert min(seg) <= -1000, "planted shock missing"


def test_validation_passed(synth):
    s, _, _, _ = synth
    p = s / "validation.json"
    if not p.exists():
        pytest.skip("run `make synth` (validate step) first")
    v = json.loads(p.read_text())
    assert not v["hard_failures"], v["hard_failures"]
    assert v["notes"]["planted_shock_recall"] == 1.0
    assert v["notes"]["lean_disagreements"] == {
        k: 0 for k in ("runs", "shocks", "drawdowns", "vrecoveries",
                       "volbursts", "assoc")}


def test_class_mix_tracks_genspec(synth, root):
    _, _, plan, meta = synth
    gs_p = root / "lean" / "out" / "genspec.json"
    gs = json.loads(gs_p.read_text())
    import hashlib
    if meta["genspec_sha256"] != hashlib.sha256(gs_p.read_bytes()).hexdigest():
        pytest.skip("synth predates the current genspec — rerun `make synth`")
    want = {c["name"]: c["w"] for c in gs["classes"]["cells"]}
    got = Counter(c for t in plan["tickers"] for c in t["classes"])
    tw, tg = sum(want.values()), sum(got.values())
    tv = 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg) for c in CLASSES)
    # generator blends transitions/dow/vol/market on top of the marginal mix;
    # the declared soft tolerance is 0.25 (validation), hard bound here 0.35
    assert tv < 0.35, f"class mix TV {tv:.3f}"


def test_certificate_kernel_checked(synth):
    s, _, _, _ = synth
    p = s / "certificate.json"
    if not p.exists():
        pytest.skip("run `make certify` first")
    cert = json.loads(p.read_text())
    assert cert["proved"] is True
    assert cert["kernel_only"] is True
