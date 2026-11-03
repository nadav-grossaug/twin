-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation
-- (their whole reservation history is one record).
-- Cell grid: the PUBLIC return-rate bands, fixed order rr_0_19..rr_80_100
-- (the share of a user's trips that revisit an earlier city of the same
-- trip — the 38% same-trip repeat-city motif, expressed per user). Every
-- user has >= 1 trip, so each user falls in exactly ONE cell: sensitivity
-- = clip = dp_count_clip = 1 (the stock mart_dp_comove pattern; no
-- proportional allocation needed).
with cells as (
    select * from (values
        ('rr_0_19', 1), ('rr_20_39', 2), ('rr_40_59', 3),
        ('rr_60_79', 4), ('rr_80_100', 5)
    ) as t(cell, ord)
),

users as (
    select userid, rr_band as cell
    from {{ ref('feat_users') }}
    where rr_band is not null
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
