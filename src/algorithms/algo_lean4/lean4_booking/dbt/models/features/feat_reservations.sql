-- Per-leg reservation features: the previous leg of the same trip (prev_*),
-- the inter-leg gap and its contiguity flag (the 99.11% temporal-flow fact),
-- and the city-return flag (this leg's city equals ANY earlier city of the
-- same trip — the 38% same-trip revisit motif). All thresholds are public
-- spec vars. Rows flagged qf_bad_dates / qf_overlap in staging are dropped
-- here, so no feature, motif or DP input ever sees them (they are counted
-- in q_data_integrity instead). Measured source: 0 flagged rows.
with s as (
    select *
    from {{ ref('stg_reservations') }}
    where not qf_bad_dates
      and not qf_overlap
),

g as (
    select
        *,
        lag(city_id)        over w as prev_city_id,
        lag(hotel_country) over w as prev_country,
        lag(hotel_ct)       over w as prev_ct,
        lag(checkout_day)   over w as prev_checkout_day,
        -- every earlier city of the same trip, in leg order (NULL on leg 1)
        list(city_id) over (
            partition by utrip order by leg_idx
            rows between unbounded preceding and 1 preceding) as earlier_cities
    from s
    window w as (partition by utrip order by leg_idx)
),

g2 as (
    select
        *,
        checkin_day - prev_checkout_day as gap_days
    from g
)

select
    * exclude (earlier_cities),
    coalesce(gap_days = 0, false)                        as is_contiguous,
    coalesce(list_contains(earlier_cities, city_id),
             false)                                      as city_return
from g2
