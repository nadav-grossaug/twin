-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation.
-- Cell grid: the 4 PUBLIC countries-per-trip bins cpt_1 / cpt_2 / cpt_3 /
-- cpt_4p (spec 1.4.0; measured 72.2 / 18.4 / 6.4 / 2.9% of trips, mean 1.41
-- countries). A user contributes their FIRST trip's count (one cell per
-- user) -> sensitivity = clip = dp_count_clip = 1. The generator draws a
-- per-trip COUNTRY BUDGET from this and locks the tier walk to the visited
-- set at budget — the unconstrained walk rendered 2.26 countries/trip vs
-- the real 1.41 (synthetic globe-trotters).
with cells as (
    select * from (values
        ('cpt_1', 1), ('cpt_2', 2), ('cpt_3', 3), ('cpt_4p', 4)
    ) as t(cell, ord)
),

users as (
    select
        t.userid,
        case t.n_countries
            when 1 then 'cpt_1'
            when 2 then 'cpt_2'
            when 3 then 'cpt_3'
            else 'cpt_4p'
        end as cell
    from (
        select
            userid,
            n_countries,
            row_number() over (
                partition by userid order by start_day, utrip) as rn
        from {{ ref('feat_trips') }}
    ) t
    where t.rn = 1
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
