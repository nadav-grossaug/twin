"""DNS name-structure layer: SQL extraction, domain mart, Lean agreement and
the char-LM minimodel.

The recompute tests re-derive label structure, RFC 1035 well-formedness and
registered domains from raw flow.jsonl in Python and compare against dbt's
SQL. Lean agreement reads out/lean_results.json (IsWellFormedQName /
IsSubdomainFan re-decided by `lake exe checks`).
"""
import json
from pathlib import Path

import pytest

from spec import load_spec

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def raw_queries():
    """(qname) for every query record in flow.jsonl, flow order preserved."""
    out = []
    with open(ROOT / "flow.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("sa") is None or not r.get("dns"):
                continue
            for d in r["dns"]:
                if d.get("qn"):
                    out.append(d["qn"])
    return out


def test_sql_query_grain_matches_raw(con, raw_queries):
    """feat_dns has one row per query record; the qname multiset matches."""
    sql = sorted(r[0] for r in con.execute("select qname from feat_dns").fetchall())
    assert sql == sorted(raw_queries)


def test_wellformed_and_domain_recompute(con, raw_queries):
    spec = load_spec()
    max_label = int(spec["dns_max_label_len"])
    max_name = int(spec["dns_max_name_len"])

    def wf(q):
        ls = q.split(".")
        return (len(q) <= max_name and len(ls) >= 1
                and all(1 <= len(l) <= max_label for l in ls))

    def dom(q):
        ls = q.split(".")
        return ".".join(ls[-2:]) if len(ls) >= 2 else q

    py_wf = sum(wf(q) for q in raw_queries)
    sql_wf = con.execute(
        "select sum(is_wellformed::int) from feat_dns").fetchone()[0]
    assert sql_wf == py_wf

    py_doms = sorted({dom(q) for q in raw_queries})
    sql_doms = sorted(r[0] for r in con.execute(
        "select distinct registered_domain from feat_dns").fetchall())
    assert sql_doms == py_doms


def test_domain_mart_partitions_queries(con):
    n_q, mart_sum = con.execute(
        "select (select count(*) from feat_dns),"
        " (select sum(n_queries) from mart_dns_domains)").fetchone()
    assert mart_sum == n_q
    fan_min = int(load_spec()["dns_subfan_min_names"])
    for n, fan in con.execute(
            "select n_distinct_names, is_subdomain_fan from mart_dns_domains").fetchall():
        assert bool(fan) == (n >= fan_min)


def test_lean_dns_name_agreement(lean_results):
    rows = lean_results.get("dns_names", [])
    assert rows, "no DNS name checks exported — run `make lean-checks`"
    bad = [r for r in rows if not r["agree"]]
    assert not bad, f"SQL/Lean IsWellFormedQName disagreement: {bad[:3]}"


def test_lean_dns_fan_agreement(lean_results):
    rows = lean_results.get("dns_fans", [])
    assert rows, "no DNS fan checks exported"
    bad = [r for r in rows if not r["agree"]]
    assert not bad, f"SQL/Lean IsSubdomainFan disagreement: {bad}"


def test_dns_lm_in_active_artifact(root):
    reg = root / "models_out" / "registry.json"
    if not reg.exists():
        pytest.skip("no model registry — run `make train`")
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not active:
        pytest.skip("no active minimodels artifact")
    vdir = root / "models_out" / "minimodels" / active
    meta = json.loads((vdir / "meta.json").read_text())
    lm = meta.get("dns_name_lm")
    assert lm, "active artifact predates the DNS char LM — retrain"
    assert (vdir / "dnslm.pt").exists()
    m = lm["metrics"]
    assert {"val_bits_per_char", "unigram_baseline_bits", "adopted", "charset"} <= set(m)
    assert m["adopted"] == (m["val_bits_per_char"] < m["unigram_baseline_bits"])


def test_dns_scores_materialized(con):
    cols = {r[1] for r in con.execute("pragma table_info('ml_flow_scores')").fetchall()}
    assert {"dns_name_score", "dns_top_qname"} <= cols
    model_version = con.execute(
        "select any_value(model_version) from ml_flow_scores").fetchone()[0]
    if model_version == "untrained":
        pytest.skip("scores materialized without an active artifact")
    n_dns, n_scored = con.execute(
        "select (select count(distinct flow_id) from feat_dns),"
        " (select count(*) from ml_flow_scores where dns_name_score is not null)"
    ).fetchone()
    assert n_scored == n_dns, "every DNS flow (and only DNS flows) gets a name score"
