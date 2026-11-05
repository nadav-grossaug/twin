"""Per-column tests of the synthetic data.

Every column of synth/flow_synth.jsonl must either
  (a) match its DECLARED distribution — the FlowShape.lean weights for the
      verified sampler draws (chi-square GOF), or the DP-released service mix;
  (b) exhibit the declared statistical/random quality (uniformity of session
      steps and beacon jitter, bounds from the linking theorems); or
  (c) satisfy its structural invariant (derived pkts, constant fields,
      address plan, port/protocol consistency).
Skips gracefully when `make synth` has not been run.
"""
import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "synth"


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def flows():
    p = SYNTH / "flow_synth.jsonl"
    if not p.exists():
        pytest.skip("no synthetic data — run `make synth`")
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert "end-config" in rows[0], "first line must be the joy config header"
    return rows[1:]


@pytest.fixture(scope="module")
def plan():
    p = SYNTH / "plan.json"
    if not p.exists():
        pytest.skip("no synth plan — run `make synth`")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def meta():
    p = SYNTH / "meta.json"
    if not p.exists():
        pytest.skip("no synth meta — run `make synth`")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def shape():
    """Generator parameters, regex-read from FlowShape.lean (the master)."""
    text = (ROOT / "lean" / "EnclaveFlows" / "FlowShape.lean").read_text()
    out = {}
    for m in re.finditer(r"def (\w+) : Nat := (\d+)", text):
        out[m[1]] = int(m[2])
    for m in re.finditer(r"def (\w+) : List Nat := \[([\d,\s]*)\]", text):
        out[m[1]] = [int(x) for x in m[2].split(",") if x.strip()]
    m = re.search(r'def genServices : List String := \[([^\]]*)\]', text)
    out["genServices"] = re.findall(r'"([^"]+)"', m[1])
    return out


@pytest.fixture(scope="module")
def beacon_channels(plan):
    return {(b["sa"], b["da"], b["dp"]) for b in plan["beacons"]}


def non_beacon(flows, beacon_channels):
    return [f for f in flows if (f["sa"], f["da"], f["dp"]) not in beacon_channels]


# ------------------------------------------------------------ chi² helper
def chi2_pvalue(observed, expected):
    """Wilson–Hilferty approximation; cells with expected < 1 are pooled out."""
    pairs = [(o, e) for o, e in zip(observed, expected) if e >= 1.0]
    dof = len(pairs) - 1
    if dof < 1:
        return 1.0
    chi2 = sum((o - e) ** 2 / e for o, e in pairs)
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def gof_against_weights(values, choices, weights, label):
    n = len(values)
    assert set(values) <= set(choices), f"{label}: values outside declared choices"
    tot = sum(weights)
    observed = [sum(1 for v in values if v == c) for c in choices]
    expected = [n * w / tot for w in weights]
    p = chi2_pvalue(observed, expected)
    assert p > 0.001, (f"{label}: χ² GOF vs declared weights fails "
                       f"(p={p:.2e}, obs={observed}, exp={[round(e,1) for e in expected]})")


# --------------------------------------------------- identifier columns
def test_sa_address_plan(flows, meta):
    # Synthetic hosts live in the 10.99.0.0/16 documentation range: host i
    # renders as 10.99.(i-1)//254 . (i-1)%254+1 — valid octets past a /24
    # (i = 1..254 keeps the historical 10.99.0.i spelling).
    sas = {f["sa"] for f in flows}
    pat = re.compile(r"10\.99\.(\d+)\.(\d+)$")
    for s in sas:
        m = pat.fullmatch(s)
        assert m, s
        third, fourth = int(m[1]), int(m[2])
        assert 0 <= third <= 255 and 1 <= fourth <= 254, s
    assert len(sas) == meta["hosts"]


def test_da_external_pool(flows, plan):
    sas = {f["sa"] for f in flows}
    for f in flows:
        assert (re.fullmatch(r"198\.51\.100\.\d+", f["da"])        # popular external head
                or re.fullmatch(r"198\.(18|19)\.\d+\.\d+", f["da"])  # long tail
                or re.fullmatch(r"10\.99\.\d+\.\d+", f["da"])), f["da"]  # internal peer
    # internal peers are real hosts, and no flow targets its own source
    assert all(f["da"] != f["sa"] for f in flows)
    assert {f["da"] for f in flows if f["da"].startswith("10.99.")} <= sas
    # topology: busy hosts fan OUT — sources are internal hosts only, externals never source
    assert all(f["sa"].startswith("10.99.") for f in flows)


def test_sp_ephemeral(flows):
    for f in flows:
        assert 49152 <= f["sp"] <= 65535
        assert f["sp"] != 53


def test_port_protocol_consistency(flows):
    svc_of_dp = {80: ("http", 6), 443: ("https", 6), 53: ("dns", 17),
                 123: ("ntp", 17), 5222: ("chat", 6), 8888: ("other", 6)}
    for f in flows:
        if f["dp"] >= 50000:
            assert f["pr"] == 6  # ephemeral service
        else:
            svc, pr = svc_of_dp[f["dp"]]
            assert f["pr"] == pr, (f["dp"], f["pr"])


# ------------------------------------- sampled columns: declared weights
def test_bytes_out_distribution(flows, beacon_channels, shape):
    vals = [f["bytes_out"] for f in non_beacon(flows, beacon_channels)]
    gof_against_weights(vals, shape["bytesOutChoices"], shape["bytesOutWeights"],
                        "bytes_out")


def test_bytes_in_distribution(flows, beacon_channels, shape):
    vals = [f["bytes_in"] for f in non_beacon(flows, beacon_channels)]
    gof_against_weights(vals, shape["bytesInChoices"], shape["bytesInWeights"],
                        "bytes_in")


def test_duration_distribution(flows, beacon_channels, shape):
    vals = [round((f["time_end"] - f["time_start"]) * 1000)
            for f in non_beacon(flows, beacon_channels)]
    gof_against_weights(vals, shape["flowDurationMsChoices"],
                        shape["flowDurationMsWeights"], "duration_ms")


def test_service_mix_matches_release(plan, shape, meta):
    # The distilled law: the FIRST flow of each session ~ released service mix;
    # later flows follow the DP transition histogram (so the flow-level mix is
    # the chain's visit distribution, not the marginal — tested upstream by
    # validate's TV check with its declared tolerance).
    rel = (meta.get("released_facts", {}).get("service_mix", {})
           .get("weights_over_genServices"))
    if not rel:
        pytest.skip("released weights not recorded in meta")
    firsts = [s for h in plan["hosts"].values()
              for s in h.get("session_first_services", [])]
    if len(firsts) < 40:
        pytest.skip("too few sessions for a GOF test")
    tot = sum(rel[s] for s in shape["genServices"])
    observed = [sum(1 for s in firsts if s == c) for c in shape["genServices"]]
    expected = [len(firsts) * rel[c] / tot for c in shape["genServices"]]
    p = chi2_pvalue(observed, expected)
    assert p > 0.001, f"first-flow service mix vs release: p={p:.2e} obs={observed}"


# ------------------------------------------- timing: bounds + uniformity
def test_session_steps_uniform(flows, beacon_channels, shape):
    """Intra-session gaps were drawn uniformly on [1, sessionStepMsMax] ms."""
    max_ms = shape["sessionStepMsMax"]
    by_host = {}
    for f in sorted(non_beacon(flows, beacon_channels),
                    key=lambda f: (f["sa"], f["time_start"])):
        by_host.setdefault(f["sa"], []).append(f["time_start"])
    steps = []
    for ts in by_host.values():
        for a, b in zip(ts, ts[1:]):
            gap_ms = (b - a) * 1000
            if gap_ms <= max_ms:  # intra-session step
                steps.append(gap_ms)
    assert steps, "no intra-session steps found"
    assert all(0 < s <= max_ms + 0.5 for s in steps)
    k = 8
    observed = [sum(1 for s in steps if i * max_ms / k < s <= (i + 1) * max_ms / k)
                for i in range(k)]
    p = chi2_pvalue(observed, [len(steps) / k] * k)
    assert p > 0.001, f"session steps not uniform on (0,{max_ms}]ms: p={p:.2e}"


def test_inter_session_gaps_break(flows, beacon_channels, shape):
    """Gaps above the step bound must exceed interSessionGapMsMin (a real break)."""
    max_ms = shape["sessionStepMsMax"]
    min_break = shape["interSessionGapMsMin"]
    by_host = {}
    for f in sorted(non_beacon(flows, beacon_channels),
                    key=lambda f: (f["sa"], f["time_start"])):
        by_host.setdefault(f["sa"], []).append(f["time_start"])
    for ts in by_host.values():
        for a, b in zip(ts, ts[1:]):
            gap_ms = (b - a) * 1000
            if gap_ms > max_ms + 0.5:
                assert gap_ms >= min_break - 0.5, gap_ms


def test_beacon_gaps_bounded_and_uniform(flows, plan, shape):
    """Beacon gaps lie in [p−j, p+j] (the genBeacon_isBeacon hypothesis) and
    the jitter is uniform on that interval (pooled χ² across channels)."""
    j = shape["beaconJitterMs"]
    # one pass to bucket channel flows by (sa, da, dp) — the per-beacon scan
    # is O(beacons × flows) and crawls at 10k hosts (5,002 × 154k)
    beacons = {(b["sa"], b["da"], b["dp"]) for b in plan["beacons"]}
    times = {}
    for f in flows:
        k = (f["sa"], f["da"], f["dp"])
        if k in beacons:
            times.setdefault(k, []).append(f["time_start"])
    pooled = []
    for b in plan["beacons"]:
        chan = sorted(times[(b["sa"], b["da"], b["dp"])])
        assert len(chan) >= shape["beaconFlowsMin"]
        p_ms = b["period_ms"]
        for a, c in zip(chan, chan[1:]):
            gap_ms = (c - a) * 1000
            assert p_ms - j - 0.5 <= gap_ms <= p_ms + j + 0.5, (b["sa"], gap_ms)
            pooled.append(gap_ms - (p_ms - j))  # normalize to [0, 2j]
    k = 6
    observed = [sum(1 for x in pooled if i * 2 * j / k <= x < (i + 1) * 2 * j / k or
                    (i == k - 1 and x >= 2 * j))
                for i in range(k)]
    p = chi2_pvalue(observed, [len(pooled) / k] * k)
    assert p > 0.001, f"beacon jitter not uniform on [p−j, p+j]: p={p:.2e}"


# --------------------------------------------------- derived + constants
def test_pkts_derived_from_bytes(flows):
    for f in flows:
        assert f["num_pkts_out"] == 1 + f["bytes_out"] // 1400
        expected_in = 0 if f["bytes_in"] == 0 else 1 + f["bytes_in"] // 1400
        assert f["num_pkts_in"] == expected_in


def test_structural_constants(flows):
    for f in flows:
        assert f["time_end"] > f["time_start"]
        for k in ("packets", "byte_dist", "entropy"):
            assert f[k] is None
        assert f["ip"] == {"out": {"ttl": 64}, "in": {"ttl": 57}}
        assert f["expire_type"] == "i"


def test_tls_dns_structural_renders(flows):
    """https flows carry a canonical-handshake tls block, dns flows carry one
    well-formed query record, everything else stays null (the mixes come from
    the DP release's aux histograms; ordinary/service-discovery names from the
    policy-released common-knowledge top-k, with a .test gibberish fallback)."""
    topk_p = ROOT / "lean" / "out" / "dns_topk.json"
    topk = ({n["qname"] for n in json.loads(topk_p.read_text())["names"]}
            if topk_p.exists() else set())
    sni_p = ROOT / "lean" / "out" / "tls_sni_topk.json"
    sni_topk = ({n["sni"] for n in json.loads(sni_p.read_text())["names"]}
                if sni_p.exists() else set())
    n_tls = n_dns = 0
    for f in flows:
        if f["dp"] == 443:
            n_tls += 1
            t = f["tls"]
            assert f["dns"] is None
            recs = t["srlt"]
            # canonical handshake by construction (IsCanonicalHandshake)
            assert recs[0]["dir"] == ">" and recs[0]["tp"] == 22
            for d in (">", "<"):
                idx = [r["tp"] for r in recs if r["dir"] == d]
                assert idx.index(20) < idx.index(23)
            assert t["cs"] and t["scs"] in t["cs"]
            # sni provenance: released top-k hostname or the synthetic fallback
            assert (t["sni"][0] in sni_topk or t["sni"] == ["svc.synth.test"])
        elif f["dp"] == 53:
            n_dns += 1
            assert f["tls"] is None
            (q,) = f["dns"]
            assert q["rc"] == 0
            labels = q["qn"].split(".")
            assert len(labels) >= 2
            assert all(1 <= len(l) <= 63 for l in labels)
            assert len(q["qn"]) <= 253
            # provenance: released top-k name, synthetic .test, or reverse lookup
            assert (q["qn"] in topk or q["qn"].endswith(".test")
                    or q["qn"].endswith(".in-addr.arpa"))
        else:
            assert f["tls"] is None and f["dns"] is None
    assert n_tls > 0
    if n_dns == 0:
        pytest.skip("this draw produced no dns-service flows (released dns share is tiny)")


def test_tls_dns_mix_sources_declared(meta):
    assert meta["genspec"]["version"]
    assert meta["generator"].startswith("lake exe generate")
    rf = meta["released_facts"]
    for k in ("service_mix", "transitions", "tls_stack_mix", "dns_shape_mix"):
        assert "genspec" in rf[k]["source"]
    assert sum(rf["tls_stack_mix"]["weights"].values()) >= 0
