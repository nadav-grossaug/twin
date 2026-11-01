-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation
-- (their whole travel history is one record; the protected fact is dataset
-- membership + trajectory).
-- Cell grid: the PUBLIC travels-per-user bins, fixed order tc_1..tc_33p.
-- Every user falls in exactly ONE cell, so the plain user count is already
-- contribution-bounded: sensitivity = clip = dp_count_clip = 1 (the stock
-- mart_dp_comove pattern; no proportional allocation needed).
with cells as (
    select * from (values
        ('tc_1', 1), ('tc_2', 2), ('tc_3_4', 3), ('tc_5_8', 4),
        ('tc_9_16', 5), ('tc_17_32', 6), ('tc_33p', 7)
    ) as t(cell, ord)
),

users as (
    select userid, tc_bin as cell
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
