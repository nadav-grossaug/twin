-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER (add/remove-one Neighbour; their whole travel
-- history is one record).
-- Cell grid: the 24 PUBLIC UTC hours of arrival day (h00..h23).
-- Contribution bounding is PROPORTIONAL (L1) over least(n_travels,
-- dp_clip_total) with largest-remainder rounding — the stock mart_dp_counts
-- semantics (see mart_dp_region for the full rationale).
with cells as (
    select * from (values
        ('h00',  0), ('h01',  1), ('h02',  2), ('h03',  3),
        ('h04',  4), ('h05',  5), ('h06',  6), ('h07',  7),
        ('h08',  8), ('h09',  9), ('h10', 10), ('h11', 11),
        ('h12', 12), ('h13', 13), ('h14', 14), ('h15', 15),
        ('h16', 16), ('h17', 17), ('h18', 18), ('h19', 19),
        ('h20', 20), ('h21', 21), ('h22', 22), ('h23', 23)
    ) as t(cell, ord)
),

events as (
    select userid, hod_cell as cell
    from {{ ref('feat_transfers') }}
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
