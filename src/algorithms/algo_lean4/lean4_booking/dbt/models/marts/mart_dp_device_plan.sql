-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation.
-- Cell grid: the 7 PUBLIC device-plan signatures (spec 1.4.0):
--   1:desktop / 1:mobile / 1:tablet  — the single device of a 1-device trip
--   2:dm / 2:dt / 2:mt              — the unordered pair of a 2-device trip
--   3:all                           — the one 3-device signature
-- A user contributes their FIRST trip's signature (one cell) -> sensitivity
-- = clip = dp_count_clip = 1. WHY: real 1-device trips are 70.9% desktop
-- (device-constant users skew desktop) while the per-leg marginal is only
-- 64.6% — drawing plans from the marginal under-rendered desktop
-- (59.7% vs the real 64.6%). Real 2-device pairs split ~51% desktop legs;
-- the 3-device share is 37.3%.
with cells as (
    select * from (values
        ('1:desktop', 1), ('1:mobile', 2), ('1:tablet', 3),
        ('2:dm', 4), ('2:dt', 5), ('2:mt', 6), ('3:all', 7)
    ) as t(cell, ord)
),

plans as (
    select
        utrip,
        string_agg(distinct device_class, ',' order by device_class) as sig
    from {{ ref('stg_reservations') }}
    group by 1
),

users as (
    select
        t.userid,
        case
            when t.n_dc = 1 then '1:' || t.sig
            when t.sig = 'desktop,mobile' then '2:dm'
            when t.sig = 'desktop,tablet' then '2:dt'
            when t.sig = 'mobile,tablet' then '2:mt'
            else '3:all'
        end as cell
    from (
        select
            p.utrip,
            p.sig,
            f.userid,
            f.n_device_classes as n_dc,
            row_number() over (
                partition by f.userid order by f.start_day, f.utrip) as rn
        from plans p
        join {{ ref('feat_trips') }} f using (utrip)
    ) t
    where t.rn = 1
)

select
    c.cell,
    coalesce(count(u.cell), 0)::bigint as clipped_count,
    count(u.cell)                     as n_contributing_users,
    {{ var('dp_count_clip') }}        as clip
from cells c
left join users u on u.cell = c.cell
group by c.cell, c.ord
order by c.ord
