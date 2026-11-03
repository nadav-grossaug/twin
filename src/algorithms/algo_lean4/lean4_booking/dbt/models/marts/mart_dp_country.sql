-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER (add/remove-one Neighbour; their whole reservation
-- history is one record).
-- Cell grid: the 16 PUBLIC hotel-country tiers (15 named countries +
-- ct_other), fixed contract order.
-- Contribution bounding is PROPORTIONAL (L1): each user's tier vector is
-- their own reservation mix allocated over a budget of least(n_res,
-- dp_clip_total) legs (largest-remainder rounding) — exactly the stock
-- mart_dp_counts semantics: users with fewer legs than the clip contribute
-- their actual counts, never inflated to the full budget. The vector sums
-- to <= dp_clip_total and no cell exceeds it, so changing one user moves
-- each cell by at most dp_clip_total — the per-cell sensitivity the Lean
-- mechanism (lake exe release) is calibrated to. (Uniform per-cell clipping
-- flattens the mix — reference-repo lesson.)
with cells as (
    select * from (values
        ('Gondal', 1), ('Cobra Island', 2), ('Fook Island', 3),
        ('Glubbdubdrib', 4), ('Elbonia', 5), ('Borginia', 6),
        ('Kangan', 7), ('Bozatta', 8), ('Alvonia', 9),
        ('Holy Britannian Empire', 10), ('Axphain', 11),
        ('Santa Prisca', 12), ('Patusan', 13), ('Rolisica', 14),
        ('Kasnia', 15), ('ct_other', 16)
    ) as t(cell, ord)
),

events as (
    select userid, hotel_ct as cell
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
