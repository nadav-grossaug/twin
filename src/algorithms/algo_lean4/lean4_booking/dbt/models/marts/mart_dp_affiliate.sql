-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER (add/remove-one Neighbour; their whole reservation
-- history is one record).
-- Cell grid: the 9 PUBLIC affiliate tiers (8 named affiliate ids +
-- aff_other), fixed contract order.
-- Contribution bounding is PROPORTIONAL (L1) over least(n_res,
-- dp_clip_total) with largest-remainder rounding — the stock mart_dp_counts
-- semantics (see mart_dp_country for the full rationale).
with cells as (
    select * from (values
        ('a9924', 1), ('a359', 2), ('a384', 3), ('a9452', 4),
        ('a4541', 5), ('a8436', 6), ('a10332', 7), ('a3631', 8),
        ('aff_other', 9)
    ) as t(cell, ord)
),

events as (
    select userid, aff_tier as cell
    from {{ ref('feat_reservations') }}
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
