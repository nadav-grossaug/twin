-- Per-trip (utrip) structure: length, city/country breadth, time span, the
-- repeat-city motif (n_cities < n_legs), the tour motif (>= tour_min_legs
-- legs spanning >= tour_min_countries distinct hotel countries with every
-- inter-leg gap <= tour_max_gap_d), trip-level contiguity (every gap = 0)
-- and the long-stay motif (some night >= longstay_min_nights).
-- ENCLAVE-ONLY in per-trip form (identifying); DP releases go through
-- mart_dp_*.
with legs as (
    select * from {{ ref('feat_reservations') }}
),

trans as (
    select
        utrip,
        count(*)                                     as n_transitions,
        max(gap_days)                                as max_gap_days,
        sum(case when gap_days = 0 then 1 else 0 end) as n_contiguous
    from legs
    where leg_idx >= 2
    group by utrip
)

select
    l.userid,
    l.utrip,
    count(*)                                                as n_legs,
    count(distinct l.city_id)                               as n_cities,
    count(distinct l.hotel_country)                         as n_countries,
    min(l.checkin_day)                                      as start_day,
    max(l.checkout_day)                                     as end_day,
    max(l.checkout_day) - min(l.checkin_day)                as span_days,
    sum(l.nights)                                           as total_nights,
    max(l.nights)                                           as nights_max,
    count(distinct l.city_id) < count(*)                    as has_repeat_city,
    max(l.nights) >= {{ var('longstay_min_nights') }}       as has_longstay,
    coalesce(t.n_transitions, 0)                            as n_transitions,
    coalesce(t.n_contiguous, 0)                             as n_contiguous,
    coalesce(t.max_gap_days, 0)                            as max_gap_days,
    -- a single-leg trip has no transition: vacuously contiguous
    coalesce(t.n_transitions > 0 and t.max_gap_days = 0,
             true)                                          as all_contiguous,
    count(distinct l.hotel_country) >= {{ var('tour_min_countries') }}
      and count(*) >= {{ var('tour_min_legs') }}
      and coalesce(t.max_gap_days <= {{ var('tour_max_gap_d') }}, true)
                                                            as is_tour,
    -- within-trip booking-channel persistence (spec 1.3.0): device and
    -- affiliate class counts per trip — real trips are mostly booked on
    -- ONE device class (70.7%) and stick to few affiliate tiers (47.5% one
    -- tier; 75.4% of legs keep the previous leg's tier). iid per-leg draws
    -- from the marginals would render far more switching.
    count(distinct l.device_class)                           as n_device_classes,
    count(distinct l.aff_tier)                               as n_aff_tiers,
    count(distinct l.device_class) = 1                       as is_devconst,
    count(distinct l.aff_tier) = 1                           as is_affconst
from legs l
left join trans t using (utrip)
group by l.userid, l.utrip, t.n_transitions, t.n_contiguous, t.max_gap_days
