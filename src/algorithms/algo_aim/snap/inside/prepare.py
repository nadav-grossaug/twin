"""Data preparation (work package G1): SNAP ego-Facebook -> ``AttributedGraph``.

Structure comes from ``facebook_combined.txt.gz``; attributes from the per-ego ``.feat``,
``.featnames`` and ``.egofeat`` files. Feature *names* are global while columns are ego-local,
so each ego's columns are mapped to names and merged (OR) into one node-by-feature matrix.
Attribute selection is data-dependent and outside the DP guarantee (see the policy file).

Run ``python snap/prepare.py`` (from the repo root, ``PYTHONPATH=src:snap``) to derive the policy,
write the report and save the processed graph.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from graph_aim.transfer.graph_io import load_attributed_graph, save_attributed_graph  # noqa: F401 (re-exported)
from graph_aim.transfer.types import AttributedGraph
from snap_loader import load_edgelist

SNAP_DIR = Path(__file__).resolve().parent
DATA_DIR = SNAP_DIR / "data"
POLICY_PATH = SNAP_DIR / "attribute_policy.json"
REPORT_PATH = SNAP_DIR / "reports" / "attributes.md"
PROCESSED_PATH = DATA_DIR / "processed" / "attributed_graph.npz"

N_NODES = 4039
DROPPED_FIELDS = ["first_name", "middle_name", "last_name", "name"]
MAX_ATTRS = 8
MAX_DOMAIN = 8  # explicit values + "other" + "none"

# Attribute specs used to derive the policy: (name, feature-name prefix, number of explicit values).
DEFAULT_SPECS = [
    ("gender", "gender", 2),
    ("locale", "locale", 4),
    ("birthday", "birthday", 5),
    ("school", "education;school;id", 4),
    ("employer", "work;employer;id", 5),
    ("location", "location;id", 5),
    ("hometown", "hometown;id", 4),
    ("language", "languages;id", 4),
]
DEFAULT_HOMOPHILY = ["school"]

NOTE = (
    "Attribute fields, top-K values and homophily attributes were chosen from the real data. "
    "This selection is data-dependent and outside the DP guarantee in v1 (a DP top-K is v2); "
    "the released domains are treated as public."
)


def field_of(name: str) -> str:
    return name.split(";", 1)[0]


def _short(name: str) -> str:
    m = re.search(r"(\d+)$", name)
    return m.group(1) if m else name


# ---------------------------------------------------------------- features


def parse_ego(
    feat_lines: list[str], featnames_lines: list[str], egofeat_line: str, ego: int
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Parse one ego network: ``(node_ids, column_names, bool matrix)``; the ego is the last row."""
    names = [ln.rstrip("\n").split(" ", 1)[1] for ln in featnames_lines if ln.strip()]
    ids: list[int] = []
    rows: list[list[int]] = []
    for ln in feat_lines:
        parts = ln.split()
        if not parts:
            continue
        ids.append(int(parts[0]))
        rows.append([int(x) for x in parts[1:]])
    ids.append(ego)
    rows.append([int(x) for x in egofeat_line.split()])
    mat = np.asarray(rows, dtype=np.int8).astype(bool)
    if mat.shape[1] != len(names):
        raise ValueError(f"ego {ego}: {mat.shape[1]} feature columns but {len(names)} names")
    return np.asarray(ids, dtype=np.int64), names, mat


def merge_egos(
    egos: list[tuple[np.ndarray, list[str], np.ndarray]], n: int
) -> tuple[list[str], np.ndarray]:
    """Merge ego-local feature matrices into one global ``(n, F)`` bool matrix (OR on overlaps)."""
    all_names = sorted({nm for _, names, _ in egos for nm in names})
    col = {nm: j for j, nm in enumerate(all_names)}
    out = np.zeros((n, len(all_names)), dtype=bool)
    for ids, names, mat in egos:
        cols = np.asarray([col[nm] for nm in names], dtype=np.int64)
        out[np.ix_(ids, cols)] |= mat
    return all_names, out


def load_features(data_dir: Path, n: int) -> tuple[list[str], np.ndarray]:
    fb = Path(data_dir) / "facebook"
    egos = []
    for fn in sorted(fb.glob("*.featnames"), key=lambda p: int(p.stem)):
        ego = int(fn.stem)
        egos.append(
            parse_ego(
                (fb / f"{ego}.feat").read_text().splitlines(),
                fn.read_text().splitlines(),
                (fb / f"{ego}.egofeat").read_text(),
                ego,
            )
        )
    if not egos:
        raise FileNotFoundError(f"no *.featnames files in {fb}")
    return merge_egos(egos, n)


def load_structure(data_dir: Path) -> np.ndarray:
    df, _ = load_edgelist(Path(data_dir) / "facebook_combined.txt.gz")
    return normalize_edges(df[["src", "dst"]].to_numpy())


def normalize_edges(raw: np.ndarray) -> np.ndarray:
    """Undirected simple edge array: ``u < v``, unique, no self loops, sorted."""
    e = np.sort(np.asarray(raw, dtype=np.int64), axis=1)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0)


# ---------------------------------------------------------------- attributes


def _prefix_cols(names: list[str], prefix: str) -> list[int]:
    return [j for j, nm in enumerate(names) if nm == prefix or nm.startswith(prefix + ";")]


def derive_policy(
    names: list[str],
    x: np.ndarray,
    specs: list[tuple[str, str, int]] = DEFAULT_SPECS,
    homophily_attrs: list[str] | None = None,
) -> dict:
    """Pick the top-K most frequent features per spec and freeze them in a policy dict."""
    counts = x.sum(axis=0)
    attrs = []
    for name, prefix, k in specs:
        cols = _prefix_cols(names, prefix)
        cols.sort(key=lambda j: (-int(counts[j]), names[j]))
        # missing_label / ordered are read by the eval metrics only (optional; defaults "none" / False)
        attrs.append(
            {
                "name": name,
                "prefix": prefix,
                "k": k,
                "values": [names[j] for j in cols[:k]],
                "missing_label": "none",
                "ordered": False,
            }
        )
    used = {field_of(a["prefix"]) for a in attrs}
    all_fields = sorted({field_of(nm) for nm in names})
    return {
        "version": 1,
        "dropped_fields": DROPPED_FIELDS,
        "fields_kept": sorted(used),
        "fields_not_used": [f for f in all_fields if f not in used and f not in DROPPED_FIELDS],
        "domain_rule": "values + 'other' + 'none'; domain size = k + 2 <= 8",
        "attributes": attrs,
        "homophily_attrs": homophily_attrs if homophily_attrs is not None else list(DEFAULT_HOMOPHILY),
        "note": NOTE,
    }


def check_policy(policy: dict) -> None:
    attrs = policy["attributes"]
    if len(attrs) > MAX_ATTRS:
        raise ValueError(f"{len(attrs)} attributes > {MAX_ATTRS}")
    names = [a["name"] for a in attrs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate attribute names")
    for a in attrs:
        if field_of(a["prefix"]) in policy["dropped_fields"]:
            raise ValueError(f"attribute {a['name']} uses a dropped identifier field")
        if a["k"] + 2 > MAX_DOMAIN or len(a["values"]) > a["k"]:
            raise ValueError(f"attribute {a['name']}: domain too large")
    if not set(policy["homophily_attrs"]) <= set(names):
        raise ValueError("homophily_attrs must be attribute names")


def homophily_groups(policy: dict) -> int:
    size = {a["name"]: a["k"] + 2 for a in policy["attributes"]}
    return int(np.prod([size[h] for h in policy["homophily_attrs"]]))


def compile_attributes(
    names: list[str], x: np.ndarray, policy: dict
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Categorical codes per node. Domain order: values (by frequency), ``other``, ``none``."""
    check_policy(policy)
    col = {nm: j for j, nm in enumerate(names)}
    counts = x.sum(axis=0)
    n = x.shape[0]
    data: dict[str, np.ndarray] = {}
    domains: dict[str, list[str]] = {}
    for a in policy["attributes"]:
        values = a["values"]
        domains[a["name"]] = [f"{a['name']}_{_short(v)}" for v in values] + ["other", "none"]
        other, none = len(values), len(values) + 1
        cols = _prefix_cols(names, a["prefix"])
        # rank by global frequency (ties by name); the node takes its highest-ranked feature
        cols.sort(key=lambda j: (-int(counts[j]), names[j]))
        code_of = {col[v]: i for i, v in enumerate(values)}
        codes = np.full(n, none, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        for j in cols:
            hit = x[:, j] & ~assigned
            codes[hit] = code_of.get(j, other)
            assigned |= hit
        data[a["name"]] = codes
    return pd.DataFrame(data, index=pd.RangeIndex(n)), domains


# ---------------------------------------------------------------- validation and I/O


def validate_attributed_graph(g: AttributedGraph, n: int | None = None) -> None:
    """Raise ``ValueError`` if the documented ``AttributedGraph`` invariants do not hold."""
    if n is not None and g.n != n:
        raise ValueError(f"n={g.n}, expected {n}")
    e = g.edges
    if e.ndim != 2 or e.shape[1] != 2 or e.dtype != np.int64:
        raise ValueError("edges must be an (m, 2) int64 array")
    if len(e):
        if e.min() < 0 or e.max() >= g.n:
            raise ValueError("edge endpoint outside 0..n-1")
        if (e[:, 0] == e[:, 1]).any():
            raise ValueError("self loop")
        if (e[:, 0] > e[:, 1]).any():
            raise ValueError("edge with u > v")
        if len(np.unique(e, axis=0)) != len(e):
            raise ValueError("duplicate edges")
    if len(g.attrs) != g.n or not g.attrs.index.equals(pd.RangeIndex(g.n)):
        raise ValueError("attrs must have index 0..n-1")
    if list(g.attrs.columns) != list(g.domains):
        raise ValueError("attrs columns and domains keys differ")
    for c, dom in g.domains.items():
        if "other" not in dom or "none" not in dom:
            raise ValueError(f"domain {c} lacks 'other' or 'none'")
        if len(set(dom)) != len(dom):
            raise ValueError(f"domain {c} has duplicate labels")
        codes = g.attrs[c].to_numpy()
        if codes.min() < 0 or codes.max() >= len(dom):
            raise ValueError(f"codes of {c} outside the domain")


# ---------------------------------------------------------------- build and report


def build_attributed_graph(
    data_dir: str | Path = DATA_DIR, policy_path: str | Path | None = POLICY_PATH
) -> AttributedGraph:
    """Build the graph; frozen values come from ``policy_path`` (derived from the data if absent)."""
    data_dir = Path(data_dir)
    names, x = load_features(data_dir, N_NODES)
    policy = _read_or_derive(names, x, policy_path)
    attrs, domains = compile_attributes(names, x, policy)
    g = AttributedGraph(n=N_NODES, edges=load_structure(data_dir), attrs=attrs, domains=domains)
    validate_attributed_graph(g, N_NODES)
    return g


def _read_or_derive(names: list[str], x: np.ndarray, policy_path: str | Path | None) -> dict:
    if policy_path is not None and Path(policy_path).exists():
        return json.loads(Path(policy_path).read_text())
    return derive_policy(names, x)


def edge_homophily(g: AttributedGraph, attr: str) -> tuple[float, float]:
    """(observed, expected-by-chance) share of edges whose endpoints share the same group."""
    codes = g.attrs[attr].to_numpy()
    obs = float((codes[g.edges[:, 0]] == codes[g.edges[:, 1]]).mean())
    deg = np.bincount(g.edges.ravel(), minlength=g.n)
    share = np.bincount(codes, weights=deg, minlength=len(g.domains[attr])) / deg.sum()
    return obs, float((share**2).sum())


def attribute_lift(g: AttributedGraph, attr: str, missing_label: str = "none") -> tuple[float, float, float, float]:
    """(lift, same-value share, chance, observed-pair share) of one attribute, excluding missing labels.

    Over the edges whose two ends are observed (neither endpoint has ``missing_label``): the share of
    edges with equal values, divided by the degree-weighted chance level ``sum_i a_i^2`` (``a_i`` =
    share of those edges' endpoints with value ``i``). ``edge_homophily`` counts missing-missing
    matches and understates the signal; this is the candidate ranking for ``homophily_groups``
    (non-private, data-dependent, outside the DP guarantee).
    """
    dom = g.domains[attr]
    codes = g.attrs[attr].to_numpy()
    cu, cv = codes[g.edges[:, 0]], codes[g.edges[:, 1]]
    keep = np.ones(len(g.edges), dtype=bool)
    if missing_label in dom:
        miss = dom.index(missing_label)
        keep = (cu != miss) & (cv != miss)
    pair_share = float(keep.mean()) if len(keep) else 0.0
    cu, cv = cu[keep], cv[keep]
    if len(cu) == 0:
        return 0.0, 0.0, 0.0, pair_share
    same = float((cu == cv).mean())
    ends = np.bincount(np.concatenate([cu, cv]), minlength=len(dom)).astype(float)
    a = ends / ends.sum()
    chance = float((a**2).sum())
    return (same / chance if chance > 0 else 0.0), same, chance, pair_share


def render_report(g: AttributedGraph, names: list[str], x: np.ndarray, policy: dict) -> str:
    cover_any = float((x.sum(axis=1) > 0).mean())
    lines = [
        "# Attribute preparation (ego-Facebook)",
        "",
        f"Nodes: {g.n}, edges: {len(g.edges)}. Nodes with at least one feature: {cover_any:.1%} "
        f"({x.shape[1]} distinct feature names merged from 10 ego networks).",
        "",
        f"Dropped identifier fields: {', '.join(policy['dropped_fields'])}.",
        f"Fields kept: {', '.join(policy['fields_kept'])}. "
        f"Fields not used: {', '.join(policy['fields_not_used'])}.",
        "",
        "| attribute | feature prefix | domain size | coverage (not none) | other | none |",
        "|---|---|---|---|---|---|",
    ]
    for a in policy["attributes"]:
        dom = g.domains[a["name"]]
        c = g.attrs[a["name"]].to_numpy()
        none = float((c == dom.index("none")).mean())
        other = float((c == dom.index("other")).mean())
        lines.append(
            f"| {a['name']} | `{a['prefix']}` | {len(dom)} | {1 - none:.1%} | {other:.1%} | {none:.1%} |"
        )
    lines += ["", "## Homophily attributes", ""]
    lines.append(
        f"Groups used for `attr_edge_pair`: {', '.join(policy['homophily_attrs'])} "
        f"({homophily_groups(policy)} groups)."
    )
    lines += ["", "Same-group edge share, observed vs chance (degree-weighted), all attributes "
              "(counts missing-missing matches, so it UNDERSTATES homophily; the ranking below does not):", ""]
    lines += ["| attribute | observed | chance |", "|---|---|---|"]
    for a in policy["attributes"]:
        o, e = edge_homophily(g, a["name"])
        lines.append(f"| {a['name']} | {o:.3f} | {e:.3f} |")
    lines += ["", "## Candidate homophily groups (non-private ranking)", ""]
    lines.append(
        "Lift per single attribute, over edges whose two ends are observed (neither endpoint has the "
        "attribute's missing label): same-value edge share divided by the degree-weighted chance level "
        "`sum_i a_i^2`. This is the data-dependent input for choosing `homophily_groups` in the config "
        "(outside the DP guarantee, see the caveat below)."
    )
    lines += ["", "| rank | attribute | lift | same-value share | chance | edges with both ends observed |",
              "|---|---|---|---|---|---|"]
    missing = {a["name"]: a.get("missing_label", "none") for a in policy["attributes"]}
    ranked = sorted(
        ((a["name"], *attribute_lift(g, a["name"], missing[a["name"]])) for a in policy["attributes"]),
        key=lambda t: -t[1],
    )
    for rank, (name, lift, same, chance, pair) in enumerate(ranked, start=1):
        lines.append(f"| {rank} | {name} | {lift:.2f} | {same:.3f} | {chance:.3f} | {pair:.1%} |")
    lines += ["", "## Caveat", "", policy["note"], ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--policy", default=str(POLICY_PATH))
    ap.add_argument("--keep-policy", action="store_true", help="reuse an existing policy file")
    args = ap.parse_args()

    names, x = load_features(Path(args.data_dir), N_NODES)
    if args.keep_policy and Path(args.policy).exists():
        policy = json.loads(Path(args.policy).read_text())
    else:
        policy = derive_policy(names, x)
        Path(args.policy).write_text(json.dumps(policy, indent=2) + "\n")
    attrs, domains = compile_attributes(names, x, policy)
    g = AttributedGraph(
        n=N_NODES, edges=load_structure(Path(args.data_dir)), attrs=attrs, domains=domains
    )
    validate_attributed_graph(g, N_NODES)
    save_attributed_graph(g, PROCESSED_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(g, names, x, policy))
    print(f"saved {PROCESSED_PATH} ({g.n} nodes, {len(g.edges)} edges)")


if __name__ == "__main__":
    main()
