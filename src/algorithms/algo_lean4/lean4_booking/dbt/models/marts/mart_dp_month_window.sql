-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation
-- (their whole reservation history is one record).
-- Cell grid: the 15 PUBLIC month-of-window cells mw00..mw14 (mw00 = the
-- window's first calendar month 2015-12; window_start_day/window_end_day
-- in the spec). Each user contributes the calendar month of their FIRST
-- reservation (one cell per user), so sensitivity = clip = dp_count_clip = 1
-- (the stock mart_dp_booker pattern).
-- WHY month-of-window and not month-of-year (spec 1.2.0): the real
-- distribution is an observation RAMP (2016-08 peak 19.5% -> 2017-02
-- 1.5%) caused by window censoring; drawing start months from a
-- month-of-year release flattened the calendar and distorted the edge
-- months by up to 15x (2017-02: real 1.5% vs synth 0.01%). This release
-- carries the real placement shape.
with cells as (
    select * from (values
        ('mw00', 0), ('mw01', 1), ('mw02', 2), ('mw03', 3), ('mw04', 4),
        ('mw05', 5), ('mw06', 6), ('mw07', 7), ('mw08', 8), ('mw09', 9),
        ('mw10', 10), ('mw11', 11), ('mw12', 12), ('mw13', 13), ('mw14', 14)
    ) as t(cell, ord)
),

users as (
    select
        userid,
        'mw' || lpad(
            cast(date_diff('month', DATE '2015-12-01',
                           date_trunc('month', min_checkin)) as varchar), 2, '0'
        ) as cell
    from (
        select userid, min(checkin) as min_checkin
        from {{ ref('stg_reservations') }}
        group by 1
    )
)

select
    c.cell,
    coalesce(count(u.userid), 0)::bigint as clipped_count,
    count(u.userid)                      as n_contributing_users,
    {{ var('dp_count_clip') }}           as clip
from cells c
left join users u on u.cell = c.cell
group by c.cell, c.ord
order by c.ord
