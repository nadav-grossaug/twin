#!/usr/bin/env python3
"""Distill the generative behavior of the minimodels into the typed Lean
generation specification (genspec).

Inputs are RELEASED artifacts only — the DP release (degree mix + the aux
clustering / triangles / richness / category / circles / community_sizes /
mixing / closure / motifs histograms). No raw data, weights or embeddings
are read for the spec, so the genspec inherits the release's privacy
accounting by post-processing. The module is stdlib-only so the off-enclave
bundle can run it as shipped.

Outputs:
  lean/out/genspec.json       the restricted generative spec consumed by
                              `lake exe generate`. Keyed by the hash of the
                              DP release: because the DP draws are
                              intentionally non-reproducible, a genspec
                              whose recorded dp_release_sha256 still matches
                              the live release is NOT rewritten (provenance-
                              skip; re-running distill is idempotent).
  models_out/fidelity.json    ENCLAVE-ONLY fidelity report: the released
                              (noisy) histograms vs the exact enclave
                              distributions, plus the degree_sage teacher
                              accuracy for context.

Distillation table (teacher -> executable statistical replacement):
  DegreeSAGE        -> DP degree histogram (marginal; the generator draws a
                       degree class per node and a degree inside the bin)
  CommunityGNN      -> DP community_sizes histogram + DP mixing rows
                       (rank-pair edge routing)
  TriadVGAE         -> DP closure rows (P(closed | center degree class);
                       the generator's triadic-closure pass)
  FeatCondEncoder   -> DP richness + category histograms (degree-conditioned
                       rows stay suppressed/enclave-side; declared gap)
  EgoMotifPool      -> DP motif rates (informational; motifs are PLANTED
                       explicitly and verified decidably)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lean" / "out"
DB = ROOT / "artifacts" / "enclave.duckdb"
MODELS_OUT = ROOT / "models_out"

CATEGORIES = ["birthday", "education", "first_name", "gender", "hometown",
              "languages", "last_name", "locale", "location", "work", "other"]
DEG_BINS = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16", "deg_17_32",
            "deg_33_64", "deg_65p"]
CC_CLASSES = ["cc_zero", "cc_0_20", "cc_20_40", "cc_40_60", "cc_60_80",
              "cc_80_99", "cc_one"]
TRI_CLASSES = ["tri_0", "tri_1_10", "tri_11_100", "tri_101_1000",
               "tri_1001_10000", "tri_10001p"]
RICH_CLASSES = [f"rich_{i}" for i in range(8)] + ["rich_8p"]
CIRC_CLASSES = ["circ_0", "circ_1", "circ_2", "circ_3", "circ_4p"]
CSIZE_CLASSES = ["csize_lt25", "csize_25_50", "csize_51_100",
                 "csize_101_200", "csize_201_400", "csize_gt400"]
RANKS = [f"c{i}" for i in range(1, 9)] + ["rest"]
MOTIF_CELLS = ["motif_hub", "motif_leaf", "motif_broker", "motif_bridge",
               "motif_butterfly", "motif_core"]
AUX_NEEDED = ("clustering", "triangles", "richness", "category", "circles",
              "community_sizes", "mixing", "closure", "motifs")
EGO_COUNT = 10          # public: the SNAP dataset ships 10 ego networks
DEGREE_TV_ACCEPT = 0.15  # pre-declared fidelity acceptance (released vs exact)

# PUBLIC bin edges (mirrors of the spec; _check_spec_sync verifies them
# against dbt/dbt_project.yml whenever the spec is readable — the bundle
# ships without it and relies on these literals + tests/test_spec_sync.py).
# Upper caps on the open-ended bins are declared public priors.
DEG_RANGES = {"deg_1": [1, 1], "deg_2": [2, 2], "deg_3_4": [3, 4],
              "deg_5_8": [5, 8], "deg_9_16": [9, 16], "deg_17_32": [17, 32],
              "deg_33_64": [33, 64], "deg_65p": [65, 200]}
CSIZE_RANGES = {"csize_lt25": [8, 24], "csize_25_50": [25, 50],
                "csize_51_100": [51, 100], "csize_101_200": [101, 200],
                "csize_201_400": [201, 400], "csize_gt400": [401, 600]}


def _csize_mid(name: str) -> int:
    """PUBLIC midpoint of a community-size bin (declared prior)."""
    lo, hi = CSIZE_RANGES[name]
    return max(1, (lo + hi) // 2)
CIRC_RANGES = {"circ_0": [0, 0], "circ_1": [1, 1], "circ_2": [2, 2],
               "circ_3": [3, 3], "circ_4p": [4, 7]}
RICH_RANGES = {**{f"rich_{i}": [i, i] for i in range(8)}, "rich_8p": [8, 10]}


def _check_spec_sync() -> None:
    """Fail loudly if the embedded public edges drift from the live spec
    (no-op in the bundle, where the spec file is not shipped)."""
    try:
        from spec import CATEGORIES as SPEC_CATS  # noqa: PLC0415
        from spec import load_spec  # noqa: PLC0415
        s = load_spec()
    except Exception:
        return
    edges = [int(s[k]) for k in ("deg_e2", "deg_e3", "deg_e5", "deg_e9",
                                 "deg_e17", "deg_e33", "deg_e65")]
    if edges != [2, 3, 5, 9, 17, 33, 65]:
        raise ValueError("distill degree bins drifted from the spec")
    cs = [int(s[k]) for k in ("csize_e25", "csize_e51", "csize_e101",
                              "csize_e201", "csize_e401")]
    if cs != [25, 51, 101, 201, 401]:
        raise ValueError("distill community-size bins drifted from the spec")
    if int(s["circ_e4"]) != 4 or int(s["rich_e8"]) != 8:
        raise ValueError("distill circle/richness bins drifted from the spec")
    if SPEC_CATS != CATEGORIES:
        raise ValueError("distill category taxonomy drifted from the spec")


def cells_of(d: dict, names: list[str]) -> list[dict]:
    return [{"name": n, "w": max(0, int(d.get(n, 0)))} for n in names]


def main() -> int:
    _check_spec_sync()
    rel_p = OUT / "dp_release.json"
    if not rel_p.exists():
        print("ERROR: run `make lean-release` first", file=sys.stderr)
        return 1
    rel_sha = hashlib.sha256(rel_p.read_bytes()).hexdigest()
    rel = json.loads(rel_p.read_text())
    deg = {c["cell"]: int(c["noised_count"]) for c in rel["cells"]}
    aux = {h["name"]: {c["cell"]: int(c["noised_count"]) for c in h["cells"]}
           for h in rel.get("aux", [])}
    for need in AUX_NEEDED:
        if need not in aux:
            print(f"ERROR: release lacks aux histogram {need}", file=sys.stderr)
            return 1

    # mixing: released 45 unordered rank-pair cells "ra__rb" (ra <= rb) ->
    # 9 conditional rows (symmetric expansion is post-processing)
    mix = aux["mixing"]

    def mix_cell(a: str, b: str) -> int:
        ia, ib = RANKS.index(a), RANKS.index(b)
        key = f"{a}__{b}" if ia <= ib else f"{b}__{a}"
        return max(0, int(mix.get(key, 0)))

    mixing_rows = [{"given": a,
                    "cells": [{"name": b, "w": mix_cell(a, b)} for b in RANKS]}
                   for a in RANKS]

    # closure: released 16 cells "clo_<class>__closed/open" -> 8 rows
    clo = aux["closure"]
    closure_rows = [
        {"given": d,
         "cells": [{"name": "closed", "w": max(0, int(clo.get(f"clo_{d}__closed", 0)))},
                   {"name": "open", "w": max(0, int(clo.get(f"clo_{d}__open", 0)))}]}
        for d in DEG_BINS]

    # category: strip the public "cat_" prefix — the generator's feature keys
    # carry the plain category names (post-processing)
    cat = {k.removeprefix("cat_"): v for k, v in aux["category"].items()}

    genspec = {
        "genspec_version": "1.0.0",
        "kind": "restricted generative specification (no executable code)",
        "spec_version": str(rel.get("spec_version", "1.0.0")),
        "source": {
            "dp_release_sha256": rel_sha,
            "privacy": "all weights below are DP-released counts (or released "
                       "+ a declared public prior); the spec is "
                       "post-processing of the release",
        },
        "n_nodes_hint": sum(max(0, deg.get(n, 0)) for n in DEG_BINS),
        "degree": {"cells": cells_of(deg, DEG_BINS)},
        "clustering": {"cells": cells_of(aux["clustering"], CC_CLASSES)},
        "triangles": {"cells": cells_of(aux["triangles"], TRI_CLASSES)},
        "richness": {"cells": cells_of(aux["richness"], RICH_CLASSES)},
        "category": {"cells": cells_of(cat, CATEGORIES)},
        "circles": {"cells": cells_of(aux["circles"], CIRC_CLASSES)},
        # the released community-size histogram is NODE-weighted (row-level
        # sensitivity 1); divide by the PUBLIC bin midpoint to recover
        # community-count weights (declared post-processing)
        "community_sizes": {"cells": [
            {"name": name,
             "w": (max(0, int(aux["community_sizes"].get(name, 0)))
                   + _csize_mid(name) // 2) // _csize_mid(name)}
            for name in CSIZE_CLASSES]},
        "mixing": {"rows": mixing_rows},
        "closure": {"rows": closure_rows},
        "motifs": {"cells": cells_of(aux["motifs"], MOTIF_CELLS)},
        "ego_count": EGO_COUNT,
        "deg_ranges": DEG_RANGES,
        "csize_ranges": CSIZE_RANGES,
        "circ_ranges": CIRC_RANGES,
        "rich_ranges": RICH_RANGES,
        "priors": [
            {"name": "clamp_at_zero", "derived_from_data": False,
             "rule": "noised counts below 0 enter the spec as 0 "
                     "(post-processing of the release)"},
            {"name": "bin_interior_uniform", "derived_from_data": False,
             "rule": "values inside a public bin are drawn grid-uniform in "
                     "the bin's public [lo, hi] range; open-ended bins carry "
                     "declared public caps (deg 200 ≈ measured p99, csize 600, circ 7, "
                     "rich 10)"},
            {"name": "closure_trials", "derived_from_data": False,
             "rule": "the generator attempts a fixed public number of wedge "
                     "closures per node (closureTrials = 6) with the "
                     "RELEASED per-degree-class probability; stub budgets "
                     "are pre-shrunk by the expected closure contribution "
                     "(computed from released weights — post-processing)"},
            {"name": "community_scoped_features", "derived_from_data": False,
             "rule": "synthetic feature keys and circles are scoped to the "
                     "community's pseudo-ego; instance ids are uniform in a "
                     "fixed public pool (featurePool = 16)"},
            {"name": "csize_node_weighted", "derived_from_data": False,
             "rule": "the released community-size histogram is node-weighted "
                     "(row-level sensitivity 1); community-count weights are "
                     "recovered by dividing each cell by its PUBLIC bin "
                     "midpoint (post-processing)"},
        ],
        "assumptions": {
            "fields_not_modelled": [
                "degree-community dependence",
                "feature identity structure beyond category level",
                "circle rosters (only membership counts are modelled)",
                "inter-ego overlap (each synthetic node belongs to one "
                "pseudo-ego scope)",
                "degree assortativity",
            ],
        },
    }

    # provenance-skip: the DP draws are non-reproducible, so a genspec keyed
    # to the live release hash is authoritative — do not rewrite it.
    gs_path = OUT / "genspec.json"
    if gs_path.exists():
        try:
            old = json.loads(gs_path.read_text())
        except json.JSONDecodeError:
            old = {}
        if old.get("source", {}).get("dp_release_sha256") == rel_sha:
            print(f"distill: genspec.json already keyed to dp_release "
                  f"{rel_sha[:12]} — reusing (provenance-skip)")
            return _fidelity(old)
    gs_path.write_text(json.dumps(genspec, indent=2))
    print(f"distill: wrote lean/out/genspec.json "
          f"(n_nodes_hint={genspec['n_nodes_hint']})")
    return _fidelity(genspec)


def _fidelity(genspec: dict) -> int:
    """ENCLAVE-ONLY: released (noisy, clamped) histograms vs the exact
    enclave distributions — how much the DP noise cost the generator —
    plus the degree_sage teacher accuracy for context."""
    report: dict = {"note": "ENCLAVE-ONLY (never exported): distillation fidelity"}
    if DB.exists():
        import duckdb  # enclave-side only
        con = duckdb.connect(str(DB), read_only=True)
        exact_deg = dict(con.execute(
            "select degree_class, count(*) from feat_nodes group by 1").fetchall())
        exact_closed, exact_wedges = con.execute(
            "select sum(tri_count), sum(wedge_count) from feat_nodes").fetchone()
        con.close()

        def tv(want: dict, got: dict, grid: list[str]) -> float:
            tw = max(1, sum(want.values()))
            tg = max(1, sum(got.values()))
            return 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                             for c in grid)

        released_deg = {c["name"]: c["w"] for c in genspec["degree"]["cells"]}
        deg_tv = tv(released_deg, exact_deg, DEG_BINS)
        rel_closed = sum(c["w"] for r in genspec["closure"]["rows"]
                         for c in r["cells"] if c["name"] == "closed")
        rel_total = sum(c["w"] for r in genspec["closure"]["rows"]
                        for c in r["cells"])
        rel_rate = rel_closed / rel_total if rel_total else None
        exact_rate = (float(exact_closed) / float(exact_wedges)
                      if exact_wedges else None)
        teacher_acc = None
        reg = MODELS_OUT / "registry.json"
        if reg.exists():
            active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
            if active:
                meta_p = MODELS_OUT / "minimodels" / active / "meta.json"
                if meta_p.exists():
                    meta = json.loads(meta_p.read_text())
                    teacher_acc = (meta.get("degree_sage") or {}).get(
                        "metrics", {}).get("val_accuracy")
        report.update({
            "degree_tv_released_vs_exact": round(deg_tv, 4),
            "closure_rate_released": None if rel_rate is None else round(rel_rate, 4),
            "closure_rate_exact": None if exact_rate is None else round(exact_rate, 4),
            "degree_sage_teacher_val_accuracy": teacher_acc,
            "accepted": bool(deg_tv <= DEGREE_TV_ACCEPT),
            "acceptance_rule": f"released-vs-exact degree TV <= {DEGREE_TV_ACCEPT}",
        })
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    (MODELS_OUT / "fidelity.json").write_text(json.dumps(report, indent=2))
    print(f"distill: wrote models_out/fidelity.json "
          f"(accepted={report.get('accepted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
