-- Return motif (repeat city): trips whose city sequence revisits a city
-- already visited earlier in the same trip — 38.07% of real trips. One row
-- per trip with the full ordered city list, checkin days, nights and
-- countries, so python/export_lean_data.py can hand Lean the same evidence
-- HasRepeatCity re-decides on. Non-repeat trips are kept too (repeat_city =
-- false) so the Lean agreement check exercises both verdicts; export caps
-- live in the export layer, not here. Matches HasRepeatCity in
-- lean/EnclaveTrip/Motifs.lean.
with seq as (
    select
        userid,
        utrip,
        list(city_id order by leg_idx)        as cities,
        list(checkin_day order by leg_idx)    as days,
        list(nights order by leg_idx)         as nights_list,
        list(hotel_country order by leg_idx)  as countries
    from {{ ref('feat_reservations') }}
    group by userid, utrip
)

select
    t.userid,
    t.utrip,
    t.n_legs,
    t.n_cities,
    s.cities,
    s.days,
    s.nights_list,
    s.countries,
    t.has_repeat_city as repeat_city
from {{ ref('feat_trips') }} t
join seq s using (userid, utrip)
order by t.n_legs desc
