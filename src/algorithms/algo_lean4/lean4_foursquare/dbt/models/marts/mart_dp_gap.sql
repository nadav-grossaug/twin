-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER (add/remove-one Neighbour; their whole travel
-- history is one record).
-- Cell grid: the 5 PUBLIC gap classes, fixed order gap_hop..gap_rare.
-- Events: a user's non-NULL inter-transfer gaps (the first transfer has no
-- previous one and contributes nothing).
-- Contribution bounding is PROPORTIONAL (L1) over least(n_gaps,
-- dp_clip_total) with largest-remainder rounding — the stock mart_dp_counts
-- semantics (see mart_dp_region for the full rationale).
with cells as (
    select * from (values
        ('gap_hop', 1), ('gap_trip', 2), ('gap_monthly', 3),
        ('gap_quarterly', 4), ('gap_rare', 5)
    ) as t(cell, ord)
),

events as (
    select userid, gap_class as cell
    from {{ ref('feat_transfers') }}
    where gap_class is not null
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
