-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation.
-- Cell grid: the 3 PUBLIC device-classes-per-trip bins dv_1 / dv_2 / dv_3
-- (spec 1.3.0; measured 70.7% / 27.6% / 1.7% of trips are booked on 1 / 2 /
-- 3 device classes). A user contributes the count of their FIRST trip (one
-- cell per user) -> sensitivity = clip = dp_count_clip = 1. The generator
-- draws a per-trip device PLAN from this: iid per-leg draws from the device
-- marginal would render ~12% single-device trips vs the real 70.7% — the
-- marginal matches, the STRUCTURE does not.
with cells as (
    select * from (values
        ('dv_1', 1), ('dv_2', 2), ('dv_3', 3)
    ) as t(cell, ord)
),

users as (
    select
        t.userid,
        case t.n_device_classes
            when 1 then 'dv_1'
            when 2 then 'dv_2'
            else 'dv_3'
        end as cell
    from (
        select
            userid,
            n_device_classes,
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
