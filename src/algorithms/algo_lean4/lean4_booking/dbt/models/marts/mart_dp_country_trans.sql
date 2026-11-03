-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER (add/remove-one Neighbour; their whole reservation
-- history is one record).
-- Cell grid: the 256 PUBLIC country-tier transition cells '<FROM>__<TO>',
-- row-major over the fixed 16-tier order (Gondal__Gondal, Gondal__Cobra
-- Island, ..., ct_other__ct_other) — INCLUDING the ct_other row/column.
-- This is the distilled replacement for the NextCountry teacher: the
-- next-destination structure the synthetic generator walks.
-- Contribution bounding is PROPORTIONAL (L1) over least(n_transitions,
-- dp_clip_total) with largest-remainder rounding — the stock mart_dp_counts
-- semantics (see mart_dp_country for the full rationale).
with tiers as (
    select * from (values
        ('Gondal', 1), ('Cobra Island', 2), ('Fook Island', 3),
        ('Glubbdubdrib', 4), ('Elbonia', 5), ('Borginia', 6),
        ('Kangan', 7), ('Bozatta', 8), ('Alvonia', 9),
        ('Holy Britannian Empire', 10), ('Axphain', 11),
        ('Santa Prisca', 12), ('Patusan', 13), ('Rolisica', 14),
        ('Kasnia', 15), ('ct_other', 16)
    ) as t(tier, ord)
),

cells as (
    select
        a.tier || '__' || b.tier as cell,
        (a.ord - 1) * 16 + b.ord as ord
    from tiers a
    cross join tiers b
),

events as (
    select userid, trans_cell as cell
    from {{ ref('feat_transitions') }}
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
