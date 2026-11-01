-- Full contingency cells for the spec'd feature pairs: observed counts over
-- the complete value grid (zero-observed cells included — chi-square is
-- wrong without them), expected counts under independence, and standardized
-- residuals. Pairs (contract): gap_class x dist_class, gap_class x
-- dur_class, is_return x gap_class, next_region x dur_class.
-- mart_feature_assoc aggregates this to per-pair statistics; the Lean layer
-- recomputes chi-square exactly over ℚ from these cells.
with base as (
    select
        obs_id,
        gap_class,
        dur_class,
        dist_class,
        next_region,
        case when is_return then 'return' else 'no_return' end as is_return
    from {{ ref('feat_transfers') }}
),

-- long format, one row per (transfer, feature); NULL values (first-transfer
-- gaps, coordinate-less distances) drop out of the unpivot
cats as (
    unpivot base
    on gap_class, dur_class, dist_class, next_region, is_return
    into name feature value value
),

-- the contract's four feature pairs (feat_a < feat_b)
wanted as (
    select * from (values
        ('dist_class', 'gap_class'),
        ('dur_class',  'gap_class'),
        ('gap_class',  'is_return'),
        ('dur_class',  'next_region')
    ) as t(feat_a, feat_b)
),

pairs as (
    select
        a.feature  as feat_a,
        b.feature  as feat_b,
        a.value    as val_a,
        b.value    as val_b,
        count(*)   as observed
    from cats a
    join cats b using (obs_id)
    join wanted w on w.feat_a = a.feature and w.feat_b = b.feature
    group by 1, 2, 3, 4
),

row_marg as (
    select feat_a, feat_b, val_a, sum(observed) as row_n
    from pairs group by 1, 2, 3
),

col_marg as (
    select feat_a, feat_b, val_b, sum(observed) as col_n
    from pairs group by 1, 2, 3
),

totals as (
    select feat_a, feat_b, sum(observed) as n
    from pairs group by 1, 2
),

grid as (
    select r.feat_a, r.feat_b, r.val_a, c.val_b, r.row_n, c.col_n, t.n
    from row_marg r
    join col_marg c using (feat_a, feat_b)
    join totals t using (feat_a, feat_b)
)

select
    g.feat_a,
    g.feat_b,
    g.val_a,
    g.val_b,
    coalesce(p.observed, 0)                    as observed,
    g.row_n,
    g.col_n,
    g.n,
    g.row_n * g.col_n / g.n::double            as expected,
    (coalesce(p.observed, 0) - g.row_n * g.col_n / g.n::double)
        / sqrt(g.row_n * g.col_n / g.n::double) as std_residual
from grid g
left join pairs p
    using (feat_a, feat_b, val_a, val_b)
