-- DP input: COLLINEARITY — per-ticker mean same-sign-day share across its
-- n-1 pairs, binned on public edges (comove_lo_pct / comove_hi_pct). Each
-- ticker contributes exactly 1 to exactly one cell, so sensitivity = clip = 1
-- (the redesign that survives n = 8: the pair-count version had sensitivity
-- n-1 = 7 and drowned in noise at any affordable ε).
with pair_shares as (
    select ticker_a as ticker, same_sign_share from {{ ref('mart_correlation') }}
    union all
    select ticker_b, same_sign_share from {{ ref('mart_correlation') }}
),

per_ticker as (
    select
        ticker,
        case
            when avg(same_sign_share) * 100 < {{ var('comove_lo_pct') }} then 'comove_lo'
            when avg(same_sign_share) * 100 < {{ var('comove_hi_pct') }} then 'comove_mid'
            else 'comove_hi'
        end as cell
    from pair_shares
    group by ticker
),

cells as (
    select unnest(['comove_lo', 'comove_mid', 'comove_hi']) as cell
)

select
    c.cell,
    coalesce(count(p.ticker), 0)::bigint   as clipped_count,
    count(p.ticker)                        as n_contributing_users,
    1                                      as clip
from cells c
left join per_ticker p using (cell)
group by c.cell
order by c.cell
