-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER (add/remove-one Neighbour; their whole travel
-- history is one record).
-- Cell grid: the 64 PUBLIC region-transition cells '<FROM>__<TO>', row-major
-- over the fixed region order (NA__NA, NA__LA, ..., OC__OC). Only seq_ok
-- rows count (gap present and <= seqbreak_gap_s — no false transitions over
-- long dormancy; a user's first transfer never counts) and 'XX' on either
-- side is filtered (tallied in q_data_integrity instead).
-- Contribution bounding is PROPORTIONAL (L1) over least(n_transitions,
-- dp_clip_total) with largest-remainder rounding — the stock mart_dp_counts
-- semantics (see mart_dp_region for the full rationale).
with regions as (
    select * from (values
        ('NA', 1), ('LA', 2), ('EU', 3), ('ME', 4),
        ('AF', 5), ('SA', 6), ('EA', 7), ('OC', 8)
    ) as t(region, ord)
),

cells as (
    select
        a.region || '__' || b.region as cell,
        (a.ord - 1) * 8 + b.ord      as ord
    from regions a
    cross join regions b
),

events as (
    select userid, trans_cell as cell
    from {{ ref('feat_transfers') }}
    where seq_ok
      and prev_region != 'XX'
      and next_region != 'XX'
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
