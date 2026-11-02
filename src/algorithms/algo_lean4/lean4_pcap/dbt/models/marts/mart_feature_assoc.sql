-- Pairwise chi-square association between all categorical features
-- (including motif flags): chi2, degrees of freedom, and Cramér's V
-- (effect size in [0,1], comparable across pairs with different table
-- sizes). The formal counterpart is EnclaveFlows/Assoc.lean, re-checked
-- on exported cells by `lake exe checks`.
select
    feat_a,
    feat_b,
    any_value(n)                                        as n,
    sum(power(observed - expected, 2) / expected)       as chi2,
    (count(distinct val_a) - 1) * (count(distinct val_b) - 1) as dof,
    count(distinct val_a)                               as card_a,
    count(distinct val_b)                               as card_b,
    sqrt(
        sum(power(observed - expected, 2) / expected)
        / (any_value(n) * (least(count(distinct val_a), count(distinct val_b)) - 1))
    )                                                   as cramers_v
from {{ ref('mart_assoc_cells') }}
group by feat_a, feat_b
-- a single-valued feature (e.g. tls_version on all-synthetic data, where no
-- flow carries TLS) has no contingency dimension: chi2 = 0 with 0 dof and
-- Cramér's V would divide by zero — exclude those pairs
having count(distinct val_a) > 1 and count(distinct val_b) > 1
order by cramers_v desc
