-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation
-- (their whole travel history is one record).
-- Cell grid: the 8 PUBLIC regions, fixed order NA..OC.
-- Contribution bounding is PROPORTIONAL (L1): each user's region vector is
-- their own region mix allocated over a budget of least(n_travels,
-- dp_clip_total) transfers (largest-remainder rounding) — exactly the stock
-- mart_dp_counts semantics: users with fewer transfers than the clip
-- contribute their actual counts, never inflated to the full budget. The
-- vector sums to <= dp_clip_total and no cell exceeds it, so changing one
-- user moves each cell by at most dp_clip_total — the per-cell sensitivity
-- the Lean mechanism (lake exe release) is calibrated to. (Uniform per-cell
-- clipping flattens the mix — reference-repo lesson.)
-- 'XX' (unknown region) never enters a DP input: filtered here, tallied in
-- q_data_integrity.
with cells as (
    select * from (values
        ('NA', 1), ('LA', 2), ('EU', 3), ('ME', 4),
        ('AF', 5), ('SA', 6), ('EA', 7), ('OC', 8)
    ) as t(cell, ord)
),

events as (
    select userid, next_region as cell
    from {{ ref('feat_transfers') }}
    where next_region != 'XX'
),

counts as (
    select userid, cell, count(*) as n
    from events
    group by userid, cell
),

totals as (
    select
        userid,
        sum(n) as total,
        least(sum(n), {{ var('dp_clip_total') }}) as budget
    from counts
    group by userid
),

alloc as (
    select
        c.userid,
        c.cell,
        (c.n * t.budget) // t.total as base,
        (c.n * t.budget) % t.total  as frac,
        t.budget
    from counts c
    join totals t using (userid)
),

ranked as (
    select
        *,
        row_number() over (partition by userid order by frac desc, cell) as rk,
        budget - sum(base) over (partition by userid) as remainder
    from alloc
),

per_user as (
    select
        userid,
        cell,
        (base + case when rk <= remainder then 1 else 0 end)::bigint as clipped_n
    from ranked
)

select
    c.cell,
    coalesce(sum(p.clipped_n), 0)::bigint                       as clipped_count,
    count(distinct case when p.clipped_n > 0 then p.userid end) as n_contributing_users,
    {{ var('dp_clip_total') }}                                  as clip
from cells c
left join per_user p on p.cell = c.cell
group by c.cell, c.ord
order by c.ord
