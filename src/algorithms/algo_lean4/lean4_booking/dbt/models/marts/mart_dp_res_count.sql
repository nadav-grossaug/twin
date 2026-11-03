-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation
-- (their whole reservation history is one record; the protected fact is
-- dataset membership + trip history).
-- Cell grid: the PUBLIC reservations-per-user bins, fixed order
-- rc_1_3..rc_13p. Every user falls in exactly ONE cell, so the plain user
-- count is already contribution-bounded: sensitivity = clip = dp_count_clip
-- = 1 (the stock mart_dp_comove pattern; no proportional allocation needed).
with cells as (
    select * from (values
        ('rc_1_3', 1), ('rc_4', 2), ('rc_5', 3), ('rc_6', 4),
        ('rc_7_8', 5), ('rc_9_12', 6), ('rc_13p', 7)
    ) as t(cell, ord)
),

users as (
    select userid, rc_bin as cell
    from {{ ref('feat_users') }}
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
