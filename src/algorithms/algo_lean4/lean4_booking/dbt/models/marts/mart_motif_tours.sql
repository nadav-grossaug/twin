-- Tour motif: trips with >= tour_min_legs legs spanning >=
-- tour_min_countries distinct hotel countries, where every inter-leg gap
-- is <= tour_max_gap_d (a fast multi-country circuit). Trip-level by
-- definition (a utrip IS the unit the challenge models). Near-miss rows
-- (is_tour = false — e.g. enough countries but one long gap) are kept so
-- the Lean agreement check exercises both verdicts; export caps live in
-- python/export_lean_data.py. Matches IsTour in Motifs.lean.
with seq as (
    select
        userid,
        utrip,
        list(checkin_day order by leg_idx)      as days,
        list(hotel_country order by leg_idx)    as countries,
        list(city_id order by leg_idx)          as cities,
        list(nights order by leg_idx)           as nights_list
    from {{ ref('feat_reservations') }}
    group by userid, utrip
)

select
    t.userid,
    t.utrip,
    s.days,
    s.countries,
    s.cities,
    s.nights_list,
    t.n_legs,
    t.n_countries,
    t.max_gap_days,
    t.n_countries >= {{ var('tour_min_countries') }}
      and t.n_legs >= {{ var('tour_min_legs') }}
      and t.max_gap_days <= {{ var('tour_max_gap_d') }} as is_tour
from {{ ref('feat_trips') }} t
join seq s using (userid, utrip)
order by t.n_countries desc, t.n_legs desc
