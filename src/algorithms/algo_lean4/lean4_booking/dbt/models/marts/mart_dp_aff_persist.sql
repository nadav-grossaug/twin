-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation.
-- Cell grid: the 4 PUBLIC affiliate-tiers-per-trip bins ap_1 / ap_2 / ap_3 /
-- ap_4p (spec 1.3.0; measured 47.5% / 39.9% / 11.2% / 1.4% of trips book
-- through 1 / 2 / 3 / 4+ affiliate tiers; 75.4% of legs keep the previous
-- leg's tier). A user contributes the count of their FIRST trip (one cell
-- per user) -> sensitivity = clip = dp_count_clip = 1. The generator draws
-- a per-trip affiliate PLAN from this; iid per-leg draws lose the strong
-- channel persistence.
with cells as (
    select * from (values
        ('ap_1', 1), ('ap_2', 2), ('ap_3', 3), ('ap_4p', 4)
    ) as t(cell, ord)
),

users as (
    select
        t.userid,
        case t.n_aff_tiers
            when 1 then 'ap_1'
            when 2 then 'ap_2'
            when 3 then 'ap_3'
            else 'ap_4p'
        end as cell
    from (
        select
            userid,
            n_aff_tiers,
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
