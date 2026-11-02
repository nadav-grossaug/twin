"""Evaluation of a synthetic table against the real train split and the real holdout.

``evaluate`` returns a flat-ish dict of metrics (scalars plus a few per-column dicts).
Every block is independent and wrapped so that one failing block records an error
string under ``<block>_error`` instead of killing the run.

Blocks:

- ``constraints``: share of rows violating each ``DatasetSpec.constraints`` rule, on the
  synthetic data and on the real holdout (the reference: the real rate is ~0).
- ``target``: target-rate preservation (positive rate for binary targets, TV distance
  of the class distribution otherwise).
- ``rare``: coverage of rare categories (< ``RARE_MIN_COUNT`` train rows) by the synthetic data.
- ``assoc``: mean absolute difference of the association matrices (train vs synthetic).
- ``marginal``: per-column KS (numeric) / total-variation (categorical) distance.
- ``utility``: TSTR vs TRTR ROC-AUC on the real holdout.
- ``privacy``: DCR (distance to closest real train record) of synthetic vs holdout rows,
  and exact-duplicate shares.
- ``synthcity``: k-anonymization and distinct l-diversity from ``synthcity.metrics``.
"""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from synth_bench.datasets import DatasetSpec

log = logging.getLogger(__name__)

RARE_MIN_COUNT = 30
DCR_MAX_ROWS = 3000  # query rows per set (synthetic, holdout); reference is the full train


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synth: pd.DataFrame,
    spec: DatasetSpec,
    seed: int = 42,
    synthcity_metrics: bool = True,
) -> dict:
    """Compute all metrics for one synthetic table. Never raises for a single bad block."""
    blocks: dict[str, Callable[[], dict]] = {
        "constraints": lambda: constraint_violations(train, test, synth, spec),
        "target": lambda: target_preservation(train, synth, spec),
        "rare": lambda: rare_category_coverage(train, synth, spec),
        "assoc": lambda: association_difference(train, synth, spec),
        "marginal": lambda: marginal_fidelity(train, synth, spec),
        "utility": lambda: utility_tstr(train, test, synth, spec, seed),
        "privacy": lambda: dcr_privacy(train, test, synth, spec, seed),
    }
    if synthcity_metrics:
        blocks["synthcity"] = lambda: synthcity_privacy(train, test, synth, spec, seed)

    out: dict = {"n_synth": len(synth)}
    for name, fn in blocks.items():
        try:
            out.update(fn())
        except Exception as exc:  # noqa: BLE001 - one metric must never kill a run
            log.warning("metric block %s failed: %s", name, exc)
            out[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- constraints


def constraint_violations(
    train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec
) -> dict:
    """Violation share per constraint on synthetic and holdout data, plus their means."""
    out: dict = {}
    synth_rates: dict[str, float] = {}
    test_rates: dict[str, float] = {}
    for name, rule in spec.constraints.items():
        synth_rates[name] = float(rule(synth).fillna(True).astype(bool).mean())
        test_rates[name] = float(rule(test).fillna(True).astype(bool).mean())
    out["constraint_violation"] = synth_rates
    out["constraint_violation_real_test"] = test_rates
    if synth_rates:
        out["constraint_violation_mean"] = float(np.mean(list(synth_rates.values())))
        out["constraint_violation_real_test_mean"] = float(np.mean(list(test_rates.values())))
        any_viol = np.zeros(len(synth), dtype=bool)
        for rule in spec.constraints.values():
            any_viol |= rule(synth).fillna(True).astype(bool).to_numpy()
        out["constraint_violation_any_row"] = float(any_viol.mean())
    return out


# --------------------------------------------------------------------------- target


def target_preservation(train: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec) -> dict:
    """Compare the target distribution of synthetic vs real train."""
    t = spec.target
    out = {"target_tv": _tv(train[t], synth[t])}
    if train[t].nunique() <= 2 and pd.api.types.is_numeric_dtype(train[t]):
        real, syn = float(train[t].mean()), float(synth[t].mean())
        out.update(target_rate_real=real, target_rate_synth=syn, target_rate_abs_diff=abs(real - syn))
    return out


# --------------------------------------------------------------------------- rare categories


def rare_category_coverage(train: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec) -> dict:
    """Share of rare train categories (< RARE_MIN_COUNT rows) that appear in synth.

    Also reports the share of synthetic categorical cells whose value never occurs in train.
    """
    rare_total = rare_hit = 0
    per_col: dict[str, float] = {}
    unseen_cells = 0
    for col in spec.categorical:
        counts = train[col].value_counts()
        rare = set(counts[counts < RARE_MIN_COUNT].index)
        present = set(synth[col].unique())
        if rare:
            hit = len(rare & present)
            per_col[col] = hit / len(rare)
            rare_total += len(rare)
            rare_hit += hit
        unseen_cells += int((~synth[col].isin(counts.index)).sum())
    n_cells = max(len(synth) * len(spec.categorical), 1)
    return {
        "rare_categories_total": rare_total,
        "rare_category_coverage": rare_hit / rare_total if rare_total else float("nan"),
        "rare_category_coverage_per_column": per_col,
        "unseen_category_cell_share": unseen_cells / n_cells,
    }


# --------------------------------------------------------------------------- association


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series (0..1)."""
    table = pd.crosstab(x, y).to_numpy()
    n = table.sum()
    r, k = table.shape
    if n == 0 or r < 2 or k < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """Correlation ratio eta (0..1) of a numeric column explained by a categorical one."""
    values = num.to_numpy(dtype=float)
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    grp = pd.Series(values).groupby(cat.to_numpy())
    between = (grp.size() * (grp.mean() - values.mean()) ** 2).sum()
    return float(np.sqrt(between / total))


def association_matrix(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Symmetric matrix of association strengths in [0, 1].

    numeric-numeric: |Pearson|; categorical-categorical: Cramér's V;
    categorical-numeric: correlation ratio. The target counts as categorical
    for classification tasks.
    """
    cats = spec.discrete_columns
    nums = [c for c in spec.numerical]
    cols = nums + cats
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if a in nums and b in nums:
                s = df[a].std() * df[b].std()
                v = abs(float(np.corrcoef(df[a], df[b])[0, 1])) if s > 0 else 0.0
            elif a in cats and b in cats:
                v = cramers_v(df[a], df[b])
            elif a in cats:
                v = correlation_ratio(df[a], df[b])
            else:
                v = correlation_ratio(df[b], df[a])
            m.loc[a, b] = m.loc[b, a] = 0.0 if np.isnan(v) else v
    return m


def association_difference(train: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec) -> dict:
    """Mean and max absolute difference of the off-diagonal association matrices."""
    diff = (association_matrix(train, spec) - association_matrix(synth, spec)).abs()
    upper = diff.to_numpy()[np.triu_indices(len(diff), k=1)]
    target_row = diff[spec.target].drop(spec.target)
    return {
        "assoc_diff_mean": float(upper.mean()),
        "assoc_diff_max": float(upper.max()),
        "assoc_diff_target_mean": float(target_row.mean()),
    }


# --------------------------------------------------------------------------- marginals


def _tv(real: pd.Series, synth: pd.Series) -> float:
    """Total-variation distance between two empirical categorical distributions."""
    p = real.value_counts(normalize=True)
    q = synth.value_counts(normalize=True)
    idx = p.index.union(q.index)
    return float(0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())


def marginal_fidelity(train: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec) -> dict:
    """Per-column KS (numeric) / TV (categorical, incl. target) distance and their means."""
    ks = {c: float(stats.ks_2samp(train[c], synth[c]).statistic) for c in spec.numerical}
    tv = {c: _tv(train[c], synth[c]) for c in spec.discrete_columns}
    both = [*ks.values(), *tv.values()]
    return {
        "marginal_ks": ks,
        "marginal_tv": tv,
        "marginal_ks_mean": float(np.mean(list(ks.values()))) if ks else float("nan"),
        "marginal_tv_mean": float(np.mean(list(tv.values()))) if tv else float("nan"),
        "marginal_mean": float(np.mean(both)),
    }


# --------------------------------------------------------------------------- utility


def _features(spec: DatasetSpec) -> list[str]:
    return [c for c in spec.columns if c != spec.target]


def _make_classifier(spec: DatasetSpec, seed: int) -> tuple[OrdinalEncoder, HistGradientBoostingClassifier]:
    cats = [c for c in spec.categorical]
    feats = _features(spec)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    mask = [c in cats for c in feats]
    clf = HistGradientBoostingClassifier(
        categorical_features=mask, max_iter=150, random_state=seed
    )
    return enc, clf


def _auc(fit_df: pd.DataFrame, score_df: pd.DataFrame, spec: DatasetSpec, seed: int) -> float:
    """ROC-AUC on ``score_df`` of a gradient boosting model fitted on ``fit_df``."""
    if fit_df[spec.target].nunique() < 2 or score_df[spec.target].nunique() < 2:
        return float("nan")
    feats, cats = _features(spec), spec.categorical
    enc, clf = _make_classifier(spec, seed)

    def prep(df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        x = df[feats].copy()
        arr = enc.fit_transform(x[cats].astype(str)) if fit else enc.transform(x[cats].astype(str))
        x[cats] = arr
        return x.astype(float)

    clf.fit(prep(fit_df, True), fit_df[spec.target])
    proba = clf.predict_proba(prep(score_df, False))
    if proba.shape[1] == 2:
        return float(roc_auc_score(score_df[spec.target], proba[:, 1]))
    return float(roc_auc_score(score_df[spec.target], proba, multi_class="ovr"))


def utility_tstr(
    train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec, seed: int
) -> dict:
    """Train on synthetic / train on real, both scored on the real holdout (ROC-AUC)."""
    tstr = _auc(synth, test, spec, seed)
    trtr = _auc(train, test, spec, seed)
    return {"tstr_auc": tstr, "trtr_auc": trtr, "tstr_auc_gap": trtr - tstr}


# --------------------------------------------------------------------------- DCR / privacy


def _gower_dcr(query: pd.DataFrame, ref: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    """Distance of each query row to its closest ref row.

    Gower-style: mean over columns of |dx|/range (numeric, range from ``ref``) or
    0/1 mismatch (categorical, target included). Result in [0, 1].
    """
    q_num = query[spec.numerical].to_numpy(dtype=np.float32)
    r_num = ref[spec.numerical].to_numpy(dtype=np.float32)
    rng = np.maximum(r_num.max(0) - r_num.min(0), 1e-9)
    cat_cols = spec.discrete_columns
    codes = {}
    for c in cat_cols:
        cat = pd.Categorical(pd.concat([ref[c], query[c]]).astype(str))
        codes[c] = (cat.codes[: len(ref)], cat.codes[len(ref) :])
    d = len(spec.numerical) + len(cat_cols)
    out = np.empty(len(query), dtype=np.float32)
    step = 256
    for s in range(0, len(query), step):
        sl = slice(s, s + step)
        acc = np.zeros((len(q_num[sl]), len(ref)), dtype=np.float32)
        for j in range(q_num.shape[1]):
            acc += np.abs(q_num[sl, j, None] - r_num[None, :, j]) / rng[j]
        for c in cat_cols:
            rc, qc = codes[c]
            acc += qc[sl, None] != rc[None, :]
        out[sl] = acc.min(1) / d
    return out


def dcr_privacy(
    train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec, seed: int
) -> dict:
    """DCR of synthetic rows vs. a holdout baseline, and exact-duplicate shares.

    ``dcr_synth_below_holdout_p5`` is the share of synthetic rows closer to train than
    the 5th percentile of the holdout DCR: ~0.05 for a generator that is as far from
    train as fresh real data, well above that for memorisation.
    """
    cols = spec.columns
    rs = np.random.RandomState(seed)

    def sample(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[rs.permutation(len(df))[:DCR_MAX_ROWS]] if len(df) > DCR_MAX_ROWS else df

    d_syn = _gower_dcr(sample(synth), train, spec)
    d_test = _gower_dcr(sample(test), train, spec)
    p5 = float(np.percentile(d_test, 5))
    train_keys = set(map(tuple, train[cols].astype(str).itertuples(index=False)))

    def dup_share(df: pd.DataFrame) -> float:
        return float(np.mean([k in train_keys for k in map(tuple, df[cols].astype(str).itertuples(index=False))]))

    out = {
        "dcr_synth_mean": float(d_syn.mean()),
        "dcr_synth_median": float(np.median(d_syn)),
        "dcr_holdout_mean": float(d_test.mean()),
        "dcr_holdout_median": float(np.median(d_test)),
        "dcr_median_ratio": float(np.median(d_syn) / max(np.median(d_test), 1e-12)),
        "dcr_synth_below_holdout_p5": float((d_syn <= p5).mean()),
        "exact_duplicate_share_synth": dup_share(synth),
        "exact_duplicate_share_holdout": dup_share(test),
        "synth_internal_duplicate_share": float(synth.duplicated().mean()),
    }
    return out


# --------------------------------------------------------------------------- synthcity


def synthcity_privacy(
    train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, spec: DatasetSpec, seed: int
) -> dict:
    """k-anonymization and distinct l-diversity from Synthcity (real train vs synth).

    Each metric is evaluated separately so that one failure does not hide the other.
    Values are the metric on real data (``.gt``) and on synthetic data (``.syn``).
    """
    from synthcity.metrics import Metrics

    from synth_bench.data import to_loader

    out: dict = {}
    gt, syn = to_loader(train, spec), to_loader(synth, spec)
    for key, name in [("k_anonymity", "k-anonymization"), ("l_diversity", "distinct l-diversity")]:
        try:
            with tempfile.TemporaryDirectory() as ws:
                res = Metrics.evaluate(
                    gt,
                    syn,
                    metrics={"privacy": [name]},
                    task_type=spec.task_type,
                    random_state=seed,
                    workspace=Path(ws),
                    use_cache=False,
                )
            for idx, row in res.iterrows():
                suffix = str(idx).rsplit(".", 1)[-1]  # gt / syn_ood / ...
                out[f"synthcity_{key}_{suffix}"] = float(row["mean"])
        except Exception as exc:  # noqa: BLE001
            out[f"synthcity_{key}_error"] = f"{type(exc).__name__}: {exc}"
    return out
