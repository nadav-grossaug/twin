"""EMPIRICAL privacy attacks on the GeoLife DP artifact and its synthetic data (eval zone).

These are NOT the differential privacy guarantee. The DP statement lives in the artifact's
privacy ledger; the checks here ask what a concrete attacker achieves against this release.
A pass does not prove privacy and a flag does not mean the accounting is wrong; they catch
gross failures (a memorising generator, a mis-set sigma) and quantify the stated risks
(seed averaging). Modelled on src/graph_aim/eval/attacks.py. Eval-only: this module may read
real data (per-user views) and must never be imported from transfer or outside.

Three attack families:

1. MEMBERSHIP INFERENCE at the user level (the DP unit), against the ARTIFACT.
   The attacker knows a target user's full data and the released noisy tables, and must decide
   whether the target was measured. Per-user contribution vectors v_{u,t} to every released
   table t (exactly the vectors measure_v2 sums before adding N(0, sigma_t^2)) give:
   - `informed_mi`: the worst case. An attacker who also knows every OTHER user's data subtracts
     them and faces y = v_u * b + noise (b in {0,1}); the likelihood-ratio statistic is Gaussian
     with separation Q_u = sum_t ||v_{u,t}||^2 / sigma_t^2, so per-user AUC = Phi(sqrt(Q_u / 2))
     and TPR at a given FPR are EXACT, no simulation needed. Q_u <= 2 * rho_total always (that is
     the zCDP bound with equality for a user who saturates every clip), so this also validates
     the accounting end to end.
   - `montecarlo_mi`: a realistic attacker who does not know the other users. Matched-filter
     score <y, v_u / sigma^2> over simulated releases (random half of the real users in, fresh
     Gaussian noise with the ledger sigmas); per-user AUC across worlds. Cross-user overlap acts
     as extra noise, so this is at most the informed AUC.

2. LINKAGE / RE-IDENTIFICATION against the SYNTHETIC data (same map frame on both sides).
   Synthetic users are fresh samples, not transformed real users, so there is no true
   correspondence; the question is whether any synthetic user is suspiciously close to a real
   one (a memorised copy). Every rate is compared against (a) a real-vs-real across-user
   baseline (chance: what an independent sample of OTHER people achieves) and (b) a within-user
   split-half reference (what a true copy of the same person would show).

3. SEED AVERAGING (`seed_averaging`): the k committed seeds are noise replicates; averaging
   them divides the noise std by sqrt(k). Measured from the cross-seed spread of the released
   tables, with the honest accounting of releasing k seeds (k * rho). Supports the
   one-seed-per-release policy (fit_artifacts.py --release).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from geolife.inside.dp import zcdp_to_epsilon
from geolife.inside.measure_v2 import UserView, make_view
from geolife.transfer import schema_v2 as S2
from geolife.transfer.artifact import Artifact
from geolife.transfer.types import UserOutings

LABEL = "EMPIRICAL DIAGNOSTIC - not the differential privacy guarantee"


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cell(c: str) -> str:
    return c[3:] if c.startswith("h3:") else c


# ============================================================ 1. membership inference


@dataclass
class UserVectors:
    """Per-user contribution vectors to every released table, in the table's released vocabulary.

    tables[name] is (n_users, n_keys); row u is exactly the vector measure_v2 sums for user u
    (pmf / one-hot: L1-normalised counts; presence: 0/1 on the user's top-m released keys).
    sigmas[name] is the ledger's noise std. Tables whose per-user views are unavailable
    (e.g. sensor tables when no sensor views were given) are listed in `skipped`.
    """
    users: list[str]
    tables: dict[str, np.ndarray]
    sigmas: dict[str, float]
    skipped: list[str] = field(default_factory=list)

    def stacked(self) -> np.ndarray:
        """(n_users, D) with every table's columns divided by its sigma (unit noise)."""
        return np.hstack([self.tables[n] / self.sigmas[n] for n in sorted(self.tables)])


def _pmf_rows(views, name: str, keys: list[tuple[str, ...]]) -> np.ndarray:
    idx = {tuple(k): i for i, k in enumerate(keys)}
    M = np.zeros((len(views), len(keys)))
    for r, v in enumerate(views):
        cnt = {idx[k]: float(n) for k, n in v.pmf.get(name, {}).items() if k in idx}
        tot = sum(cnt.values())
        if tot > 0:
            for i, x in cnt.items():
                M[r, i] = x / tot
    return M


def _presence_rows(views: list[UserView], name: str, keys: list[tuple[str, ...]]) -> np.ndarray:
    tiers, attr = (S2.PLACE_TIERS, "place") if name in S2.PLACE_TIERS else (S2.TRAV_TIERS, "trav")
    level, m = tiers.index(name), S2.max_keys_of(name)
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(views), len(keys)))
    for r, v in enumerate(views):
        cand = [(c, n) for c, n in getattr(v, attr)[level].items() if (c,) in idx]
        for c, _ in sorted(cand, key=lambda cn: (-cn[1], cn[0]))[:m]:
            M[r, idx[(c,)]] = 1.0
    return M


def build_user_vectors(users: list[UserOutings], artifact: Artifact, vocab,
                       sensor_views=None, seed: int = 0) -> UserVectors:
    """Reconstruct each user's contribution to every table of `artifact` (released keys as the
    fixed vocabulary; the hierarchical key selection is post-processing of the noisy parents, so
    taking the released keys as given matches the attacker's knowledge). `vocab` is the
    measure_v2.VocabV2 the artifact was built with; `sensor_views` (SensorView list, same
    pseudonyms) adds the sensor_* tables, else they are skipped."""
    views = [make_view(u, vocab, seed) for u in users]
    sv = {v.user: v for v in (sensor_views or [])}
    uv = UserVectors([u.user for u in users], {}, {})
    for m in artifact.measurements:
        keys = [tuple(k) for k in m.keys]
        kind = S2.KIND.get(m.name)
        if kind == "presence":
            uv.tables[m.name] = _presence_rows(views, m.name, keys)
        elif m.name in S2.SENSOR_NAMES:
            if not sv:
                uv.skipped.append(m.name)
                continue
            rows = [sv.get(u.user) for u in users]
            missing = type("E", (), {"pmf": {}})()
            uv.tables[m.name] = _pmf_rows([r if r is not None else missing for r in rows],
                                          m.name, keys)
        elif kind in ("pmf", "onehot"):
            uv.tables[m.name] = _pmf_rows(views, m.name, keys)
        else:
            uv.skipped.append(m.name)
            continue
        uv.sigmas[m.name] = float(m.sigma)
    return uv


def informed_mi(uv: UserVectors, rho_total: float | None = None,
                fprs: tuple[float, ...] = (0.01, 0.1)) -> dict:
    """EXACT per-user membership-inference strength for the informed (worst-case) attacker.

    Q_u = sum_t ||v_{u,t}||^2 / sigma_t^2; AUC_u = Phi(sqrt(Q_u / 2));
    TPR at FPR alpha = Phi(sqrt(Q_u) - z_{1-alpha}). The zCDP bound gives Q_u <= 2 * rho_total.
    """
    from scipy.stats import norm
    X = uv.stacked()
    q = (X ** 2).sum(axis=1)
    auc = np.array([_phi(math.sqrt(x / 2.0)) for x in q])
    out = {
        "label": LABEL,
        "n_users": len(uv.users),
        "tables_used": sorted(uv.tables),
        "tables_skipped": sorted(uv.skipped),
        "q": {"min": float(q.min()), "p50": float(np.median(q)), "p90": float(np.percentile(q, 90)),
              "max": float(q.max()), "bound_2rho": (2.0 * rho_total if rho_total else None)},
        "auc": {"p50": float(np.median(auc)), "p90": float(np.percentile(auc, 90)),
                "max": float(auc.max()),
                "bound": (_phi(math.sqrt(rho_total)) if rho_total else None)},
    }
    worst = math.sqrt(q.max())
    out["tpr_at_fpr_worst_user"] = {str(a): float(norm.cdf(worst - norm.ppf(1 - a))) for a in fprs}
    out["q_per_table_worst_user"] = {
        n: float(((uv.tables[n][int(q.argmax())] / uv.sigmas[n]) ** 2).sum())
        for n in sorted(uv.tables)}
    return out


def montecarlo_mi(uv: UserVectors, n_worlds: int = 400, seed: int = 0,
                  include_prob: float = 0.5) -> dict:
    """Realistic matched-filter attacker over simulated releases of the REAL mechanism.

    Each world: every user independently in with `include_prob`; release y = sum_in v + unit
    noise (vectors pre-divided by sigma). Attacker score for user u: <y, v_u>. Per-user AUC
    across worlds (in-worlds vs out-worlds); the other users' vectors act as additional noise,
    so this is the uninformed attacker and is upper-bounded by informed_mi.
    """
    rng = np.random.default_rng(seed)
    X = uv.stacked()
    n = X.shape[0]
    masks = rng.random((n_worlds, n)) < include_prob
    noise = rng.standard_normal((n_worlds, X.shape[1]))
    scores = (masks @ X + noise) @ X.T                      # (n_worlds, n_users)
    aucs = np.full(n, np.nan)
    for u in range(n):
        s_in, s_out = scores[masks[:, u], u], scores[~masks[:, u], u]
        if len(s_in) and len(s_out):
            gt = (s_in[:, None] > s_out[None, :]).mean()
            eq = (s_in[:, None] == s_out[None, :]).mean()
            aucs[u] = gt + 0.5 * eq
    ok = aucs[np.isfinite(aucs)]
    return {"label": LABEL, "n_worlds": n_worlds, "include_prob": include_prob,
            "auc": {"p50": float(np.median(ok)), "p90": float(np.percentile(ok, 90)),
                    "max": float(ok.max()), "mean": float(ok.mean())}}


# ============================================================ 2. linkage / re-identification


def _profile(u: UserOutings) -> np.ndarray:
    """Map-relative numeric fingerprint: log sizes, revisit rate, median outing shape, 6 time-bin
    shares (TOD_3 x daytype)."""
    os_ = u.outings
    days = {(o.t_start + S2.LOCAL_UTC_OFFSET_H * 3600) // 86400 for o in os_}
    places = {v.place_id for o in os_ for v in o.visits}
    seen: set = set()
    tot = rev = 0
    for o in os_:
        for v in o.visits:
            if v.kind == "stop":
                tot += 1
                rev += _cell(v.cell) in seen
        seen |= {_cell(v.cell) for v in o.visits}
    tb = Counter(S2.tod3_of(o.t_start) for o in os_)
    shares = [tb[(t, d)] / max(1, len(os_)) for t in S2.TOD_3 for d in S2.DAYTYPES]
    med = np.median
    return np.array([math.log1p(len(os_)), math.log1p(len(days)), math.log1p(len(places)),
                     rev / tot if tot else 0.0,
                     math.log1p(med([o.length_m for o in os_]) / 1000.0),
                     math.log1p(med([(o.t_end - o.t_start) / 60.0 for o in os_])),
                     med([sum(v.kind == "stop" for v in o.visits) for o in os_]), *shares])


def _top_cells(outings, k: int = 10, res: int = 8) -> frozenset:
    c = Counter()
    for o in outings:
        for v in o.visits:
            c[_cell(v.cell)] += 1
    return frozenset(c2 for c2, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k])


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _split_half(u: UserOutings, rng) -> tuple[list, list]:
    idx = rng.permutation(len(u.outings))
    h = len(u.outings) // 2
    return [u.outings[i] for i in idx[:h]], [u.outings[i] for i in idx[h:]]


def nearest_user_distance(real: list[UserOutings], syn: list[UserOutings],
                          seed: int = 0, ratio_tol: float = 0.5) -> dict:
    """Distance from each real user to the nearest synthetic user in profile space, against a
    real-vs-real holdout (random user halves A -> B). ratio_median near or above 1: the synthetic
    users are no closer than an independent sample of other people; well below 1: copy-like."""
    from scipy.spatial import cKDTree
    R = np.array([_profile(u) for u in real])
    S = np.array([_profile(u) for u in syn])
    mu, sd = R.mean(axis=0), R.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Rz, Sz = (R - mu) / sd, (S - mu) / sd
    d_syn = cKDTree(Sz).query(Rz)[0]
    perm = np.random.default_rng(seed).permutation(len(real))
    h = len(real) // 2
    a, b = perm[:h], perm[h:]
    d_hold = cKDTree(Rz[b]).query(Rz[a])[0]
    med_s, med_h = float(np.median(d_syn)), float(np.median(d_hold))
    ratio = med_s / med_h if med_h > 0 else float("nan")
    return {"label": LABEL, "n_real": len(real), "n_syn": len(syn),
            "synthetic": {"median": med_s, "p05": float(np.percentile(d_syn, 5)),
                          "min": float(d_syn.min())},
            "holdout": {"median": med_h, "p05": float(np.percentile(d_hold, 5)),
                        "min": float(d_hold.min())},
            "ratio_median": float(ratio),
            "too_close": bool(ratio == ratio and ratio < ratio_tol)}


def top_cell_linkage(real: list[UserOutings], syn: list[UserOutings], k: int = 10,
                     seed: int = 0) -> dict:
    """Can a real user be re-identified by their top-k visited res-8 cells? Both sides must be in
    the SAME map frame (e.g. generated into Beijing). For each real user: best Jaccard against
    all synthetic users, against the OTHER real users (chance: different people share hubs), and
    against their own other outing-half (the copy reference: what finding the same person looks
    like). reid_rate = share of real users whose best synthetic match is at least as similar as
    their own split half; chance_rate = the same with other real users in place of synthetic."""
    rng = np.random.default_rng(seed)
    rc = [_top_cells(u.outings, k) for u in real]
    sc = [_top_cells(u.outings, k) for u in syn]
    halves = [(_top_cells(a, k), _top_cells(b, k)) for a, b in (_split_half(u, rng) for u in real)]
    self_j = np.array([_jaccard(a, b) for a, b in halves])
    best_syn = np.array([max((_jaccard(r, s) for s in sc), default=0.0) for r in rc])
    best_other = np.array([max((_jaccard(r, o) for j, o in enumerate(rc) if j != i), default=0.0)
                           for i, r in enumerate(rc)])
    # one-to-one assignment (Hungarian) vs a random pairing
    from scipy.optimize import linear_sum_assignment
    J = np.array([[_jaccard(r, s) for s in sc] for r in rc])
    ri, si = linear_sum_assignment(-J)
    matched = float(J[ri, si].mean())
    rand = float(np.mean([J[i, j] for i, j in zip(range(len(rc)),
                                                  rng.permutation(len(sc))[:len(rc)])]))
    return {"label": LABEL, "k": k, "n_real": len(real), "n_syn": len(syn),
            "best_jaccard_syn": {"mean": float(best_syn.mean()), "p90": float(np.percentile(best_syn, 90)),
                                 "max": float(best_syn.max())},
            "best_jaccard_other_real": {"mean": float(best_other.mean()),
                                        "p90": float(np.percentile(best_other, 90)),
                                        "max": float(best_other.max())},
            "self_split_half": {"mean": float(self_j.mean()), "p50": float(np.median(self_j))},
            "reid_rate_syn": float((best_syn >= self_j).mean()),
            "reid_rate_chance": float((best_other >= self_j).mean()),
            "hungarian_mean_jaccard": matched, "random_matching_mean_jaccard": rand}


def signature_linkage(real: list[UserOutings], syn: list[UserOutings], seed: int = 0) -> dict:
    """Coarse released-histogram signature (outings, distinct places, span, revisit buckets):
    share of real users UNIQUE by signature whose signature occurs among the synthetic users,
    against a real-half holdout (unique in A, occurs in B). Matching a coarse signature is
    expected for a faithful generator; only a large excess over the holdout is suspicious."""
    def sig(u: UserOutings):
        os_ = u.outings
        span = (max(o.t_end for o in os_) - min(o.t_start for o in os_)) / 86400.0 if os_ else 0.0
        places = {_cell(v.cell) for o in os_ for v in o.visits}
        seen: set = set()
        tot = rev = 0
        for o in os_:
            for v in o.visits:
                if v.kind == "stop":
                    tot += 1
                    rev += _cell(v.cell) in seen
            seen |= {_cell(v.cell) for v in o.visits}
        return (S2.bin_of(len(os_), S2.OUTINGS_EDGES), S2.bin_of(len(places), S2.PLACES_EDGES),
                S2.bin_of(span, S2.SPAN_DAYS_EDGES),
                S2.bin_of(rev / tot if tot else 0.0, S2.REVISIT_EDGES))

    def rate(query: list, ref: list) -> tuple[float, int]:
        cnt = Counter(query)
        uniq = [s for s in query if cnt[s] == 1]
        refset = set(ref)
        return (sum(s in refset for s in uniq) / len(uniq) if uniq else float("nan"), len(uniq))

    rs, ss = [sig(u) for u in real], [sig(u) for u in syn]
    r_syn, n_uniq = rate(rs, ss)
    perm = np.random.default_rng(seed).permutation(len(real))
    h = len(real) // 2
    ra = [rs[i] for i in perm[:h]]
    rb = [rs[i] for i in perm[h:]]
    r_hold, n_uniq_h = rate(ra, rb)
    return {"label": LABEL, "n_unique_real": n_uniq, "rate_syn": float(r_syn),
            "n_unique_holdout": n_uniq_h, "rate_holdout": float(r_hold),
            "excess": float(r_syn - r_hold)}


# ============================================================ 3. seed averaging


def seed_averaging(artifacts: list[Artifact], delta: float = 1e-6) -> dict:
    """Quantify the risk of releasing several noise seeds of the same fit.

    Per common table: the cross-seed per-key std estimates sigma (replicate check); the average
    of k seeds has noise std sigma / sqrt(k), so the per-table signal-to-noise `signal
    (= |released total mass| / (sigma sqrt(K)), the fit threshold)` grows by sqrt(k). The honest
    accounting of k released seeds is k * rho_total, reported as epsilon(delta); the informed-MI
    separation Q scales by k, so the worst-case per-user AUC Phi(sqrt(Q/2)) grows accordingly.
    """
    k = len(artifacts)
    led = artifacts[0].ledger
    sig_of = {e["name"]: float(e["sigma"]) for e in led.entries}
    rows = []
    for m0 in artifacts[0].measurements:
        series = []
        for a in artifacts:
            m = next((x for x in a.measurements if x.name == m0.name), None)
            if m is None or [tuple(x) for x in m.keys] != [tuple(x) for x in m0.keys]:
                series = []
                break
            series.append(np.asarray(m.values, float))
        if len(series) < 2:
            continue
        V = np.vstack(series)
        sigma = sig_of[m0.name]
        emp = float(V.std(axis=0, ddof=1).mean())
        mass = float(np.abs(V.mean(axis=0).sum()))
        kk = V.shape[1]
        snr1 = mass / (sigma * math.sqrt(kk))
        rows.append({"table": m0.name, "n_keys": kk, "sigma_ledger": sigma,
                     "cross_seed_std": emp, "std_ratio": emp / sigma,
                     "signal_single_seed": snr1, "signal_avg_k": snr1 * math.sqrt(k),
                     "noise_std_avg_k": sigma / math.sqrt(k)})
    rho1 = led.rho_total
    return {"label": LABEL, "n_seeds": k, "tables": rows,
            "rho_single": rho1, "epsilon_single": zcdp_to_epsilon(rho1, delta),
            "rho_k_seeds": k * rho1, "epsilon_k_seeds": zcdp_to_epsilon(k * rho1, delta),
            "noise_reduction_factor": math.sqrt(k),
            "note": "informed-MI Q scales by k: worst-case per-user AUC Phi(sqrt(k*Q/2))"}
