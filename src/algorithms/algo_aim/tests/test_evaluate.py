import numpy as np
import pandas as pd

from synth_bench import evaluate as ev
from tests.conftest import make_df


def shuffled_independent(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Marginals kept, dependence destroyed (column-wise permutation)."""
    rs = np.random.RandomState(seed)
    return pd.DataFrame({c: rs.permutation(df[c].to_numpy()) for c in df.columns})


def test_constraints_real_is_zero_and_broken_is_detected(spec, splits):
    train, test = splits
    broken = test.copy()
    broken["code"] = 1
    out = ev.constraint_violations(train, test, broken, spec)
    assert out["constraint_violation_real_test"]["code_mapping"] == 0
    assert 0.4 < out["constraint_violation"]["code_mapping"] < 0.6
    assert out["constraint_violation"]["z_negative"] == 0
    assert out["constraint_violation_any_row"] == out["constraint_violation"]["code_mapping"]


def test_target_preservation(spec, splits):
    train, _ = splits
    flipped = train.assign(y=1 - train["y"])
    assert ev.target_preservation(train, train, spec)["target_rate_abs_diff"] == 0
    diff = ev.target_preservation(train, flipped, spec)
    assert np.isclose(diff["target_rate_abs_diff"], abs(1 - 2 * train["y"].mean()))


def test_rare_category_coverage(spec):
    small = make_df(300, 5)  # "rare" has ~9 rows, below the threshold
    out = ev.rare_category_coverage(small, small[small["color"] != "rare"], spec)
    assert out["rare_categories_total"] >= 1
    assert out["rare_category_coverage"] == 0
    assert ev.rare_category_coverage(small, small, spec)["rare_category_coverage"] == 1
    unseen = small.assign(color="purple")
    assert ev.rare_category_coverage(small, unseen, spec)["unseen_category_cell_share"] == 1


def test_association_measures():
    rs = np.random.RandomState(0)
    a = pd.Series(rs.choice(list("abc"), 2000))
    assert ev.cramers_v(a, a) > 0.95
    assert ev.cramers_v(a, pd.Series(rs.choice(list("abc"), 2000))) < 0.1
    num = a.map({"a": 0.0, "b": 5.0, "c": 10.0}) + rs.normal(size=2000) * 0.1
    assert ev.correlation_ratio(a, num) > 0.99
    assert ev.correlation_ratio(a, pd.Series(rs.normal(size=2000))) < 0.1


def test_association_difference_detects_lost_dependence(spec, splits):
    train, _ = splits
    assert ev.association_difference(train, train, spec)["assoc_diff_mean"] == 0
    shuffled = shuffled_independent(train)
    assert ev.association_difference(train, shuffled, spec)["assoc_diff_mean"] > 0.05


def test_marginal_fidelity(spec, splits):
    train, _ = splits
    assert ev.marginal_fidelity(train, shuffled_independent(train), spec)["marginal_mean"] < 1e-9
    far = train.assign(x=train["x"] + 10, color="red")
    out = ev.marginal_fidelity(train, far, spec)
    assert out["marginal_ks"]["x"] == 1
    assert out["marginal_tv"]["color"] > 0.4


def test_tstr_good_vs_inverted_labels(spec, splits):
    train, test = splits
    good = ev.utility_tstr(train, test, make_df(1000, 3), spec, seed=0)
    assert good["trtr_auc"] > 0.8 and good["tstr_auc"] > 0.8
    bad = ev.utility_tstr(train, test, train.assign(y=1 - train["y"]), spec, seed=0)
    assert bad["tstr_auc"] < 0.3
    const = ev.utility_tstr(train, test, train.assign(y=0), spec, seed=0)
    assert np.isnan(const["tstr_auc"])


def test_dcr_copy_vs_fresh(spec, splits):
    train, test = splits
    copy = ev.dcr_privacy(train, test, train.copy(), spec, seed=0)
    assert copy["dcr_synth_mean"] == 0
    assert copy["exact_duplicate_share_synth"] == 1
    assert copy["dcr_synth_below_holdout_p5"] == 1
    fresh = ev.dcr_privacy(train, test, make_df(500, 9), spec, seed=0)
    assert fresh["exact_duplicate_share_synth"] == 0
    assert 0.3 < fresh["dcr_median_ratio"] < 3


def test_evaluate_isolates_failing_blocks(spec, splits):
    train, test = splits
    broken = train.drop(columns=["z"])  # most blocks need z
    out = ev.evaluate(train, test, broken, spec, synthcity_metrics=False)
    assert out["n_synth"] == len(broken)
    assert any(k.endswith("_error") for k in out)
    assert "target_rate_abs_diff" in out  # blocks not needing z still produced results


def test_evaluate_full_without_synthcity(spec, splits):
    train, test = splits
    out = ev.evaluate(train, test, make_df(800, 4), spec, synthcity_metrics=False)
    assert not [k for k in out if k.endswith("_error")]
    for key in ["constraint_violation_mean", "assoc_diff_mean", "marginal_mean", "tstr_auc",
                "dcr_synth_median", "target_rate_abs_diff", "rare_category_coverage"]:
        assert key in out
