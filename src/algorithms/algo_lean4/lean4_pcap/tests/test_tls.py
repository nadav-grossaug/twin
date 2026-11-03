"""TLS fingerprinting layer: SQL feature extraction, marts, Lean agreement
and the TLS record minimodel.

The fingerprint recompute test re-derives the JA3-style hash from the raw
flow.jsonl in Python and compares against dbt's SQL (md5 over the same
canonical string) — locking the two extractions together. Lean agreement
reads out/lean_results.json (IsCanonicalHandshake / IsRareStack re-decided
per flow by `lake exe checks`).
"""
import hashlib
import json
from pathlib import Path

import pytest

from spec import load_spec

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def raw_tls():
    """flow.jsonl TLS records keyed like stg_flows orders them."""
    recs = []
    with open(ROOT / "flow.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("sa") is not None and r.get("tls"):
                recs.append(r)
    return recs


def _fp_string(tls: dict) -> str:
    cv = tls.get("c_version")
    cs = tls.get("cs") or []
    exts = sorted(k for e in (tls.get("c_extensions") or []) for k in e)
    return f"v{cv if cv is not None else '?'},{'-'.join(cs)},{'-'.join(exts)}"


def test_sql_fingerprint_matches_python_recompute(con, raw_tls):
    """Multiset of fingerprints from SQL == recompute from raw JSON."""
    sql_fps = sorted(r[0] for r in con.execute(
        "select fingerprint from feat_tls").fetchall())
    py_fps = sorted(
        hashlib.md5(_fp_string(r["tls"]).encode()).hexdigest()[:8]
        for r in raw_tls)
    assert sql_fps == py_fps


def test_canonical_handshake_sql_matches_python(con, raw_tls):
    """Per-flow canonical-handshake verdicts: SQL == independent recompute."""
    def canonical(srlt):
        if not srlt or not (srlt[0].get("dir") == ">" and srlt[0].get("tp") == 22):
            return False
        for d in (">", "<"):
            seq = [r for r in srlt if r.get("dir") == d]
            tps = [r.get("tp") for r in seq]
            if 23 in tps and (20 not in tps or tps.index(20) > tps.index(23)):
                return False
        return True

    py_count = sum(canonical(r["tls"].get("srlt") or []) for r in raw_tls)
    sql_count = con.execute(
        "select count(*) from feat_tls where is_canonical_hs").fetchone()[0]
    assert sql_count == py_count
    assert con.execute("select count(*) from feat_tls").fetchone()[0] == len(raw_tls)


def test_fingerprint_mart_partitions_flows(con):
    n_flows, mart_sum = con.execute(
        "select (select count(*) from feat_tls),"
        " (select sum(n_flows) from mart_tls_fingerprints)").fetchone()
    assert mart_sum == n_flows
    rare_max = int(load_spec()["tls_rare_max_flows"])
    for n, rare in con.execute(
            "select n_flows, is_rare_stack from mart_tls_fingerprints").fetchall():
        assert bool(rare) == (n <= rare_max)


def test_record_mart_covers_all_tls_flows(con):
    n_flows, shape_sum = con.execute(
        "select (select count(*) from feat_tls),"
        " (select sum(n_flows) from mart_tls_records)").fetchone()
    assert shape_sum == n_flows


def test_lean_tls_canonical_agreement(lean_results):
    rows = lean_results.get("tls_canonical", [])
    assert rows, "no TLS canonical checks exported — run `make lean-checks`"
    bad = [r for r in rows if not r["agree"]]
    assert not bad, f"SQL/Lean IsCanonicalHandshake disagreement: {bad[:3]}"


def test_lean_tls_rare_stack_agreement(lean_results):
    rows = lean_results.get("tls_rare", [])
    assert rows, "no TLS rare-stack checks exported"
    bad = [r for r in rows if not r["agree"]]
    assert not bad, f"SQL/Lean IsRareStack disagreement: {bad}"
    assert lean_results.get("tls_partition_ok") is True


def test_tls_minimodel_in_active_artifact(root):
    reg = root / "models_out" / "registry.json"
    if not reg.exists():
        pytest.skip("no model registry — run `make train`")
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not active:
        pytest.skip("no active minimodels artifact")
    vdir = root / "models_out" / "minimodels" / active
    meta = json.loads((vdir / "meta.json").read_text())
    tls = meta.get("tls_fingerprint")
    assert tls, "active artifact predates the TLS record model — retrain"
    assert (vdir / "tlsrecord.pt").exists()
    m = tls["metrics"]
    assert {"val_accuracy", "majority_baseline", "adopted", "classes"} <= set(m)
    assert m["adopted"] == (m["val_accuracy"] > m["majority_baseline"])


def test_tls_scores_materialized(con):
    cols = {r[1] for r in con.execute("pragma table_info('ml_flow_scores')").fetchall()}
    assert {"pred_tls_stack", "pred_tls_confidence"} <= cols
    n_tls, n_scored = con.execute(
        "select (select count(*) from feat_tls),"
        " (select count(*) from ml_flow_scores where pred_tls_stack is not null)"
    ).fetchone()
    model_version = con.execute(
        "select any_value(model_version) from ml_flow_scores").fetchone()[0]
    if model_version == "untrained":
        pytest.skip("scores materialized without an active artifact")
    assert n_scored == n_tls, "every TLS flow (and only TLS flows) gets a stack prediction"
