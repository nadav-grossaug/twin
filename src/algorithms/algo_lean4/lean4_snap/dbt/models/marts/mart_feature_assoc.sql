-- Pairwise chi-square association between the spec'd discretized features:
-- chi2, degrees of freedom, and Cramér's V per grid. The formal counterpart
-- is EnclaveGraph/Assoc.lean, re-checked on exported cells by
-- `lake exe checks`.
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
having count(distinct val_a) > 1 and count(distinct val_b) > 1
order by cramers_v desc
