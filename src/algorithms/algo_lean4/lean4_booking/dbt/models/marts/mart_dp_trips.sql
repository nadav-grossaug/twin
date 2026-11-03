-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation.
-- Cell grid: the 3 PUBLIC trips-per-user bins tp_1 / tp_2 / tp_3p
-- (spec 1.3.0; measured 91.1% / 7.4% / 1.5%). Each user falls in exactly
-- ONE cell -> sensitivity = clip = dp_count_clip = 1. The generator
-- renders MULTI-TRIP users from this (1.11 trips/user real; the
-- pre-1.3.0 single_trip prior rendered exactly 1.00).
with cells as (
    select * from (values
        ('tp_1', 1), ('tp_2', 2), ('tp_3p', 3)
    ) as t(cell, ord)
),

users as (
    select userid, tp_bin as cell
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
