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
# affiliation (education/work key-group) release: the DP member-weighted
# size histogram + the suppression-thresholded cohesion rollup
AFFIL_CATS_REL = ["birthday", "education", "gender", "hometown", "languages",
                  "last_name", "locale", "location", "work"]   # == spec.py AFFIL_RELEASE_CATS
AFFIL_BINS = ["g_2_4", "g_5_8", "g_9_16", "g_17p"]
# PUBLIC inclusive size ranges for the three closed bins (shared by every
# category). The OPEN bin g_17p gets a per-category cap: cap_c = 2*mean_c - 17,
# so a grid-uniform draw on [17, cap_c] has the measured mean group size of that
# category's >=17 groups (birthday 53.8, education 83.4, gender 214.5, hometown
# 130.5, languages 52.1, last_name 42.8, locale 213.0, location 51.9, work 32.9).
# The caps are declared public priors; the real tail is far heavier (education
# p99 623, max 823), which a uniform draw cannot express - declared, not hidden.
AFFIL_GROUP_RANGES = {"g_2_4": [2, 4], "g_5_8": [5, 8], "g_9_16": [9, 16]}
AFFIL_G17P_CAP = {"birthday": 91, "education": 150, "gender": 412,
                  "hometown": 244, "languages": 87, "last_name": 69,
                  "locale": 409, "location": 87, "work": 49}


def _affil_range(cat: str, b: str) -> list[int]:
    return [17, AFFIL_G17P_CAP[cat]] if b == "g_17p" else AFFIL_GROUP_RANGES[b]


def _affil_mid(cat: str, b: str) -> int:
    """PUBLIC midpoint of an affiliation group-size bin for a category
    (declared prior)."""
    lo, hi = _affil_range(cat, b)
    return max(1, (lo + hi) // 2)


def _affil_group_ranges() -> dict:
    """genspec group_ranges: the closed bins + g_17p default, plus an explicit
    `<category>/g_17p` override per category (the generator looks up
    `<category>/<bin>` first)."""
    out = {**AFFIL_GROUP_RANGES, "g_17p": [17, 120]}
    for c in AFFIL_CATS_REL:
        out[f"{c}/g_17p"] = _affil_range(c, "g_17p")
    return out


# the teacher's category order (train_minimodels AFFIL_CATS) for fidelity
AFFIL_TEACHER_CATS = ["education", "work", "hometown", "location",
                      "languages"]
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
    af = [int(s[k]) for k in ("affil_g5", "affil_g9", "affil_g17")]
    if af != [5, 9, 17]:
        raise ValueError("distill affiliation size bins drifted from the spec")
    try:
        from spec import AFFIL_RELEASE_CATS  # noqa: PLC0415
        if AFFIL_RELEASE_CATS != AFFIL_CATS_REL:
            raise ValueError(
                "distill affiliation categories drifted from the spec")
    except ImportError:
        pass
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

    # affiliation (education/work key-group) structure: OPTIONAL — present
    # only when the release carries the affil_sizes aux histogram AND the
    # suppression-thresholded cohesion rollup was materialized. Absent, the
    # generator keeps the declared feature-identity gap.
    affiliation = None
    affil_p = OUT / "affil_cohesion.json"
    if "affil_sizes" in aux and affil_p.exists():
        asz = aux["affil_sizes"]
        coh = json.loads(affil_p.read_text())
        dens = {(r["category"], r["size_bin"]): int(r["density_pct"])
                for r in coh["rows"]}
        size_rows, density_rows = [], []
        for c in AFFIL_CATS_REL:
            cells, dcells = [], []
            for b in AFFIL_BINS:
                w = max(0, int(asz.get(f"aff_{c}_{b}", 0)))
                if (c, b) in dens:
                    p = min(100, max(0, dens[(c, b)]))
                    dcells.append({"name": b, "w": p})
                    # the release is MEMBER-weighted; divide by the PUBLIC
                    # bin midpoint to recover group-count weights (prior
                    # affil_member_weighted, same as csize_node_weighted)
                    mid = _affil_mid(c, b)
                    w = (w + mid // 2) // mid
                else:
                    # suppressed cohesion cell -> the bin is not generated
                    # (prior affil_suppressed_bins_skipped)
                    w = 0
                cells.append({"name": b, "w": w})
            size_rows.append({"given": c, "cells": cells})
            density_rows.append({"given": c, "cells": dcells})
        affiliation = {
            "categories": AFFIL_CATS_REL,
            "size_rows": size_rows,
            "density_rows": density_rows,
            "group_ranges": _affil_group_ranges(),
            "density_den": 100,
            "source": {"sizes": "dp aux affil_sizes (member-weighted)",
                       "densities": "artifact affil_cohesion_stats "
                                    "(suppression-thresholded, no formal "
                                    "guarantee; integer percent)"},
        }

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
        **({"affiliation": affiliation} if affiliation else {}),
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
            {"name": "affil_member_weighted", "derived_from_data": False,
             "rule": "the released affiliation size histogram is "
                     "member-weighted (row-level sensitivity); group-count "
                     "weights are recovered by dividing each cell by its "
                     "PUBLIC bin midpoint (post-processing); the generator "
                     "recovers the member budget as sum(w * midpoint)"},
            {"name": "affil_window", "derived_from_data": False,
             "rule": "group members are drawn from a window of factor x group "
                     "size inside the community, factor by size bin: 3 (sizes "
                     "2-4), 16 (sizes 5-16), 8 (17+) - public generator "
                     "constants affilWindowSmall/Mid/Large. Tight windows share "
                     "near-clique block edges and OVERSHOOT the released "
                     "density; wide ones leave the degree-capped fill no room "
                     "and UNDERSHOOT; the crossover differs by size (measured, "
                     "5 runs per setting: mean |density error| 0.058 vs "
                     "0.095-0.119 for a single factor 8)"},
            {"name": "affil_stochastic_round", "derived_from_data": False,
             "rule": "a group's target edge count = floor(density * pairs) + 1 "
                     "with probability equal to the fractional part (verified "
                     "sampler). Deterministic rounding is biased on tiny "
                     "groups: a size-2 group has one pair, so any released "
                     "density >= 0.5 became a full pair"},
            {"name": "hub_full_deficit", "derived_from_data": False,
             "rule": "a node whose degree target is in the open degree bin "
                     "(>= 65, the public spec edge deg_e65) initiates its FULL "
                     "remaining degree deficit as mixing stubs; all other nodes "
                     "initiate half. Stub partners are uniform, so with half "
                     "the deficit hubs ended at ~half their target. Measured "
                     "over 8 runs each, before -> after (real): share of nodes "
                     "with degree >= 100 0.078 -> 0.105 (0.122), transitivity "
                     "0.584 -> 0.531 (0.519), modularity 0.860 -> 0.840 "
                     "(0.835), core share 0.137 -> 0.102 (0.091); edge count "
                     "unchanged. The open bin's tail beyond the declared cap "
                     "of 200 is NOT recoverable from the single released 65+ "
                     "cell"},
            {"name": "affil_closure_bias", "derived_from_data": False,
             "rule": "the density fill weights each missing in-group pair by "
                     "1 + 16 * min(common neighbours, 3) (public generator "
                     "constants affilClosureBoost = 16, affilClosureCap = 3), "
                     "normalised so the expected number of added edges is "
                     "unchanged: same released density, but the added edges "
                     "close wedges instead of opening them (per-node closure "
                     "0.491 -> 0.519 vs the 0.516 no-affiliation baseline; "
                     "degree TV unchanged)"},
            {"name": "affil_group_sample", "derived_from_data": False,
             "rule": "synthetic key-groups are distinct node samples "
                     "inside ONE community (the key_community_mlp teacher "
                     "verdict: affiliations align with structural "
                     "communities at 73.4% vs 17.8% majority); member counts "
                     "are drawn from the RELEASED member-weighted size "
                     "histogram, sizes grid-uniform in the public bin ranges "
                     "(the open bin's per-category cap 2*mean-17 matches each "
                     "category's measured mean group size, but the real tail "
                     "is heavier: education p99 623); "
                     "non-group affiliation features "
                     "carry node-unique instance ids so key-group structure "
                     "comes only from planted groups"},
            {"name": "affil_density_fill", "derived_from_data": False,
             "rule": "within-group wiring adds edges only up to the RELEASED "
                     "integer-percent density (deficit fill over missing "
                     "pairs via the verified sampler); existing block/stub "
                     "edges count toward the target, receiver degree caps "
                     "apply"},
            {"name": "affil_suppressed_bins_skipped",
             "derived_from_data": False,
             "rule": "affiliation bins whose cohesion cell was suppressed "
                     "by the release threshold are not generated (their "
                     "size weight enters the spec as 0)"},
        ],
        "assumptions": {
            "fields_not_modelled": [
                "degree-community dependence",
                ("feature identity structure beyond category level for "
                 "non-affiliation categories (education/work key-groups ARE "
                 "modelled from the released size/cohesion tables)")
                if affiliation else
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


def _affil_teacher_pred(cat: str, size_mid: float) -> float | None:
    """key_cohesion_mlp density prediction for one public (category,
    size-midpoint) cell — manual 2-layer forward so distill never imports
    the heavy training stack. Returns None off-enclave or without torch."""
    reg = MODELS_OUT / "registry.json"
    if not reg.exists():
        return None
    try:
        import math  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not active:
        return None
    pt = MODELS_OUT / "minimodels" / active / "key_cohesion_mlp.pt"
    if not pt.exists():
        return None
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    if cat not in AFFIL_TEACHER_CATS:
        return None      # the teacher was not trained on this category
    onehot = [1.0 if c == cat else 0.0 for c in AFFIL_TEACHER_CATS]
    x = torch.tensor(onehot + [math.log2(size_mid)], dtype=torch.float32)
    h = torch.relu(sd["net.0.weight"] @ x + sd["net.0.bias"])
    y = sd["net.2.weight"] @ h + sd["net.2.bias"]
    return float(y.item())


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
        # affiliation: distilled student (the released density table) vs the
        # key_cohesion_mlp teacher vs the train-mean baseline, on the public
        # (category x size-bin) grid — the teacher certifies the conditional,
        # the released values supply the weights (METHOD §8).
        if "affiliation" in genspec:
            con2 = duckdb.connect(str(DB), read_only=True)
            _cat_sql = ",".join(repr(c) for c in AFFIL_CATS_REL)
            exact_rows = {(c, b): float(d) for c, b, d in con2.execute(
                "select category, size_bin, avg(density) from ("
                "  select category, case"
                "    when n_nodes < 5 then 'g_2_4'"
                "    when n_nodes < 9 then 'g_5_8'"
                "    when n_nodes < 17 then 'g_9_16'"
                "    else 'g_17p' end as size_bin, density"
                "  from mart_key_groups where category in (" + _cat_sql + ")"
                ") group by 1, 2").fetchall()}
            mean_density = con2.execute(
                "select avg(density) from mart_key_groups "
                "where category in (" + _cat_sql + ")").fetchone()[0]
            con2.close()
            dens = {(r["given"], c["name"]): c["w"] / 100
                    for r in genspec["affiliation"]["density_rows"]
                    for c in r["cells"]}
            ranges = genspec["affiliation"]["group_ranges"]
            cells_rep, s_err, t_err, b_err = [], [], [], []
            s_err_t, b_err_t = [], []   # restricted to teacher-covered cells
            for (cat_name, b), student in sorted(dens.items()):
                exact = exact_rows.get((cat_name, b))
                if exact is None:
                    continue
                r = ranges.get(f"{cat_name}/{b}", ranges[b])
                mid = (r[0] + r[1]) / 2
                teacher = _affil_teacher_pred(cat_name, mid)
                cells_rep.append({
                    "cell": f"{cat_name}/{b}", "exact": round(exact, 4),
                    "released": round(student, 4),
                    "teacher": None if teacher is None else round(teacher, 4)})
                s_err.append(abs(student - exact))
                b_err.append(abs(float(mean_density) - exact))
                if teacher is not None:
                    t_err.append(abs(teacher - exact))
                    s_err_t.append(abs(student - exact))
                    b_err_t.append(abs(float(mean_density) - exact))

            def _mae(x):
                return round(sum(x) / len(x), 4) if x else None
            report["affiliation"] = {
                "cells": cells_rep,
                # like-for-like on the cells the teacher was trained on
                # (education, work, hometown, location, languages)
                "teacher_covered_cells": len(t_err),
                "teacher_mae": _mae(t_err),
                "student_mae_on_teacher_cells": _mae(s_err_t),
                "mean_baseline_mae_on_teacher_cells": _mae(b_err_t),
                # over every released cell (the teacher does not cover all)
                "student_mae": _mae(s_err),
                "mean_baseline_mae": _mae(b_err),
                "note": "student = released density table; teacher = "
                        "key_cohesion_mlp at public bin midpoints (5 of the "
                        "released categories); baseline = enclave mean density",
            }
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    (MODELS_OUT / "fidelity.json").write_text(json.dumps(report, indent=2))
    print(f"distill: wrote models_out/fidelity.json "
          f"(accepted={report.get('accepted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
