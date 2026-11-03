-- Full contingency cells for the spec'd feature pairs: observed counts over
-- the complete value grid (zero-observed cells included — chi-square is
-- wrong without them), expected counts under independence, and standardized
-- residuals. Pairs span three grains (steps, visits, user-days); each pair
-- is computed at its own grain, then shares the margin/expected machinery.
-- mart_feature_assoc aggregates this to per-pair statistics; the Lean layer
-- recomputes chi-square exactly over ℚ from these cells.
{% set assoc_pairs = [
    ('mode_class', 'speed_class', 'feat_steps',     "mode_class != 'XX'"),
    ('mode_class', 'hour_bin',    'feat_steps',     "mode_class != 'XX'"),
    ('role',       'arr_bin',     'feat_visits',    "true"),
    ('role',       'dwell_class', 'feat_visits',    "not arr_censored and not dep_censored"),
    ('motif_class','day_type',    'feat_user_days', "true"),
    ('mode_class', 'dist_class',  'feat_legs',      "mode_class != 'XX'"),
    ('mode_class', 'circ_class',  'feat_route_legs',"mode_class != 'XX'"),
] %}

with pairs as (
    {%- for a, b, tbl, cond in assoc_pairs %}
    select
        '{{ a }}'        as feat_a,
        '{{ b }}'        as feat_b,
        {{ a }}::varchar as val_a,
        {{ b }}::varchar as val_b,
        count(*)         as observed
    from {{ ref(tbl) }}
    where ({{ cond }}) and {{ a }} is not null and {{ b }} is not null
    group by 3, 4
    {{ "union all" if not loop.last }}
    {%- endfor %}
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
