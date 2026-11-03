-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation
-- (their whole reservation history is one record).
-- Cell grid: the 5 PUBLIC booker countries (anonymized origin taxonomy),
-- fixed order. feat_users carries each user's single booker_country (the
-- first-seen; only 531 of 259,159 users ever appear from two, so this is
-- the modal value), so every user falls in exactly ONE cell and the plain
-- user count is already contribution-bounded: sensitivity = clip =
-- dp_count_clip = 1 (the stock mart_dp_comove pattern; no proportional
-- allocation needed).
with cells as (
    select * from (values
        ('Gondal', 1), ('The Devilfire Empire', 2), ('Elbonia', 3),
        ('Tcherkistan', 4), ('Bartovia', 5)
    ) as t(cell, ord)
),

users as (
    select userid, booker_country as cell
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
