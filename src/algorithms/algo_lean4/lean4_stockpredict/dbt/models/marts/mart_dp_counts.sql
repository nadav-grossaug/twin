-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one TICKER = one "user" in SampCert's add/remove-one
-- Neighbour relation (its whole series is one record; the protected fact is
-- portfolio membership + trajectory).
-- Cell grid: the PUBLIC 5-class return taxonomy.
-- Contribution bounding is PROPORTIONAL (L1): each ticker's class vector is
-- its own class mix allocated over a total budget of dp_clip_total days
-- (largest-remainder rounding; tickers with fewer days contribute their
-- actual counts). The vector sums to <= dp_clip_total and no cell exceeds
-- it, so changing one ticker moves each cell by at most dp_clip_total — the
-- per-cell sensitivity the Lean mechanism (lake exe release) is calibrated
-- to. (Uniform per-cell clipping flattens the mix — reference-repo lesson.)
with cells as (
    select unnest(['crash', 'down', 'flat', 'up', 'surge']) as cell
),

days as (
    select ticker, ret_class as cell
    from {{ ref('feat_returns') }}
    where ret_class is not null
),

counts as (
    select ticker, cell, count(*) as n
    from days
    group by ticker, cell
),

totals as (
    select
        ticker,
        sum(n) as total,
        least(sum(n), {{ var('dp_clip_total') }}) as budget
    from counts
    group by ticker
),

alloc as (
    select
        c.ticker,
        c.cell,
        (c.n * t.budget) // t.total as base,
        (c.n * t.budget) % t.total  as frac,
        t.budget
    from counts c
    join totals t using (ticker)
),

ranked as (
    select
        *,
        row_number() over (partition by ticker order by frac desc, cell) as rk,
        budget - sum(base) over (partition by ticker) as remainder
    from alloc
),

per_user as (
    select
        ticker,
        cell,
        (base + case when rk <= remainder then 1 else 0 end)::bigint as clipped_n
    from ranked
)

select
    c.cell,
    coalesce(sum(p.clipped_n), 0)::bigint  as clipped_count,
    count(distinct case when p.clipped_n > 0 then p.ticker end) as n_contributing_users,
    {{ var('dp_clip_total') }}             as clip
from cells c
left join per_user p on p.cell = c.cell
group by c.cell
order by c.cell
