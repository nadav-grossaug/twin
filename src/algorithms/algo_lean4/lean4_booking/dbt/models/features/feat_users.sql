-- Per-user profile: reservation volume on the public rc bins, trip count,
-- city breadth, booker country (one per user in the source), and the
-- repeat-city return rate over the user's trips with its public band.
-- Every user has >= 1 trip, so rr / rr_band are always defined (unlike the
-- travel build, which needed >= 2 travels).
-- ENCLAVE-ONLY (per-user rows are identifying); DP releases go through
-- mart_dp_*.
with res as (
    select userid, utrip, city_id, booker_country
    from {{ ref('feat_reservations') }}
),

trips as (
    select userid, utrip, has_repeat_city
    from {{ ref('feat_trips') }}
),

u as (
    select
        userid,
        count(*)                as n_res,
        count(distinct utrip)   as n_trips,
        count(distinct city_id) as n_cities,
        min(booker_country)     as booker_country
    from res
    group by userid
),

r as (
    select
        userid,
        count(*)                                            as n_trips_r,
        sum(case when has_repeat_city then 1 else 0 end)    as n_repeat_trips
    from trips
    group by userid
)

select
    u.userid,
    u.n_res,
    u.n_trips,
    u.n_cities,
    case
        when u.n_res < {{ var('rc_e4') }}  then 'rc_1_3'
        when u.n_res < {{ var('rc_e5') }}  then 'rc_4'
        when u.n_res < {{ var('rc_e6') }}  then 'rc_5'
        when u.n_res < {{ var('rc_e7') }}  then 'rc_6'
        when u.n_res < {{ var('rc_e9') }}  then 'rc_7_8'
        when u.n_res < {{ var('rc_e13') }} then 'rc_9_12'
        else 'rc_13p'
    end                                       as rc_bin,
    u.booker_country,
    100.0 * r.n_repeat_trips / r.n_trips_r    as rr,
    case
        when 100.0 * r.n_repeat_trips / r.n_trips_r <     {{ var('rr_band_width_pct') }} then 'rr_0_19'
        when 100.0 * r.n_repeat_trips / r.n_trips_r < 2 * {{ var('rr_band_width_pct') }} then 'rr_20_39'
        when 100.0 * r.n_repeat_trips / r.n_trips_r < 3 * {{ var('rr_band_width_pct') }} then 'rr_40_59'
        when 100.0 * r.n_repeat_trips / r.n_trips_r < 4 * {{ var('rr_band_width_pct') }} then 'rr_60_79'
        else 'rr_80_100'
    end                                       as rr_band,
    -- trips-per-user bin (spec 1.3.0): 91.1% / 7.4% / 1.5% measured for
    -- 1 / 2 / 3+; the generator renders multi-trip users from this
    case
        when u.n_trips = 1 then 'tp_1'
        when u.n_trips = 2 then 'tp_2'
        else 'tp_3p'
    end                                       as tp_bin
from u
join r using (userid)
