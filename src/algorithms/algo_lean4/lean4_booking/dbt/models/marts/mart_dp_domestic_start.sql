-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation.
-- Cell grid: the 5 PUBLIC booker countries (spec 1.4.0). A user contributes
-- ONE cell — their booker country — if their FIRST trip's FIRST leg is
-- DOMESTIC (hotel_country = booker_country), else no cell; sensitivity =
-- clip = dp_count_clip = 1. Combined with the booker release (users per
-- booker country) this yields the per-booker domestic-start probability:
-- measured 18.7% (Gondal) / 2.1% (Devilfire) / 13.7% (Elbonia) / 5.9%
-- (Tcherkistan) / 17.5% (Bartovia). The generator drew start tiers
-- independently of booker, overstating the international rate (94.8% vs
-- the real 90.5%).
with cells as (
    select * from (values
        ('Gondal', 1), ('The Devilfire Empire', 2), ('Elbonia', 3),
        ('Tcherkistan', 4), ('Bartovia', 5)
    ) as t(cell, ord)
),

users as (
    select
        s.booker_country as cell
    from (
        select
            booker_country,
            hotel_country,
            row_number() over (
                partition by userid
                order by checkin, utrip, leg_idx) as rn
        from {{ ref('stg_reservations') }}
    ) s
    where s.rn = 1
      and s.hotel_country = s.booker_country
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
