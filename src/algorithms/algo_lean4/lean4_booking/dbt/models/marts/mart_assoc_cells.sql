-- Full contingency cells for the spec'd association grids (reservation
-- level): observed counts, expected counts under independence, and
-- standardized residuals.
-- Grids (wiki/booking-contract.md):
--   device_class x booker_country  (booking channel vs origin)
--   device_class x hotel_ct        (booking channel vs destination tier)
--   aff_tier     x device_class    (acquisition channel vs device)
--   aff_tier     x booker_country  (acquisition channel vs origin)
--   booker_country x hotel_ct       (origin vs destination — the
--                                    domestic/international structure)
--   month_cell   x hotel_ct        (seasonality vs destination tier)
-- mart_feature_assoc aggregates this to per-pair chi2 / dof / Cramér's V;
-- the Lean layer (EnclaveTrip.Assoc) recomputes chi-square exactly over ℚ
-- from these cells.
with pairs as (
    select
        'device_class'   as feat_a,
        'booker_country' as feat_b,
        device_class     as val_a,
        booker_country   as val_b,
        count(*)         as observed
    from {{ ref('feat_reservations') }}
    group by 3, 4

    union all

    select
        'device_class',
        'hotel_ct',
        device_class,
        hotel_ct,
        count(*)
    from {{ ref('feat_reservations') }}
    group by 3, 4

    union all

    select
        'aff_tier',
        'device_class',
        aff_tier,
        device_class,
        count(*)
    from {{ ref('feat_reservations') }}
    group by 3, 4

    union all

    select
        'aff_tier',
        'booker_country',
        aff_tier,
        booker_country,
        count(*)
    from {{ ref('feat_reservations') }}
    group by 3, 4

    union all

    select
        'booker_country',
        'hotel_ct',
        booker_country,
        hotel_ct,
        count(*)
    from {{ ref('feat_reservations') }}
    group by 3, 4

    union all

    select
        'month_cell',
        'hotel_ct',
        month_cell,
        hotel_ct,
        count(*)
    from {{ ref('feat_reservations') }}
    group by 3, 4
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
