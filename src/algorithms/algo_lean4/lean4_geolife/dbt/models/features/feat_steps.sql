-- Per-point kinematics within a (user, trajectory) stream: step interval,
-- haversine step distance, speed, heading and turning angle, plus the public
-- speed class and 45-degree turn bin. All thresholds are public spec vars.
-- Heading uses a local equirectangular frame (dx east, dy north, meters);
-- turning angles are defined only when this step and the previous one both
-- actually moved (>= min_move_m).
with p as (
    select
        obs_id, user_id, trajectory_id, sequence_uuid, ts_s, datetime,
        latitude, longitude, mode_class,
        lag(ts_s)      over w as prev_ts_s,
        lag(latitude)  over w as prev_lat,
        lag(longitude) over w as prev_lon
    from {{ ref('stg_points') }}
    window w as (partition by user_id, trajectory_id order by ts_s, obs_id)
),

s as (
    select
        *,
        ts_s - prev_ts_s as dt_s,
        -- haversine step distance in meters (6371000 = mean Earth radius)
        2 * 6371000.0 * asin(least(1.0, sqrt(
            pow(sin(radians(latitude - prev_lat) / 2), 2)
            + cos(radians(prev_lat)) * cos(radians(latitude))
              * pow(sin(radians(longitude - prev_lon) / 2), 2)
        )))              as dist_m,
        -- local ENU displacement for heading
        (longitude - prev_lon) * cos(radians(latitude)) * 111320.0 as dx_m,
        (latitude - prev_lat) * 111320.0                            as dy_m
    from p
    where prev_ts_s is not null
      and ts_s > prev_ts_s
),

h as (
    select
        *,
        dist_m / dt_s as speed_mps,
        case when dist_m >= {{ var('min_move_m') }}
             then degrees(atan2(dx_m, dy_m)) end as heading_deg
    from s
),

t as (
    select
        *,
        heading_deg - lag(heading_deg) over (
            partition by user_id, trajectory_id order by ts_s, obs_id
        ) as turn_raw
    from h
)

select
    obs_id,
    user_id,
    trajectory_id,
    sequence_uuid,
    ts_s,
    datetime,
    latitude,
    longitude,
    mode_class,
    dt_s,
    dist_m,
    speed_mps,
    heading_deg,
    -- normalize the turning angle to (-180, 180]
    case when turn_raw is null then null
         when turn_raw > 180  then turn_raw - 360
         when turn_raw <= -180 then turn_raw + 360
         else turn_raw end                                        as turn_deg,
    case
        when speed_mps * 1000 < {{ var('spd_still_max_mmps') }} then 'spd_still'
        when speed_mps * 1000 < {{ var('spd_walk_max_mmps') }}  then 'spd_walk'
        when speed_mps * 1000 < {{ var('spd_city_max_mmps') }}  then 'spd_city'
        when speed_mps * 1000 < {{ var('spd_fast_max_mmps') }}  then 'spd_fast'
        else 'spd_vfast'
    end                                                           as speed_class,
    case when turn_raw is not null then
        'tb_' || least(floor((
            case when turn_raw > 180 then turn_raw - 360
                 when turn_raw <= -180 then turn_raw + 360
                 else turn_raw end + 180.0) / 45.0), 7)::integer::varchar
    end                                                           as turn_bin,
    hour(datetime)::smallint                                      as hour_local,
    case
        when hour(datetime) <= {{ var('dep_night_max_h') }}     then 'dep_night'
        when hour(datetime) <= {{ var('dep_morning_max_h') }}   then 'dep_morning'
        when hour(datetime) <= {{ var('dep_afternoon_max_h') }} then 'dep_afternoon'
        else 'dep_evening'
    end                                                           as hour_bin,
    (isodow(datetime) - 1)::smallint                              as dow,  -- 0=Mon..6=Sun
    isodow(datetime) >= 6                                         as is_weekend,
    dist_m >= {{ var('min_move_m') }}
      and speed_mps * 1000 >= {{ var('spd_still_max_mmps') }}     as is_moving,
    speed_mps > {{ var('max_speed_mps') }}                        as qf_implausible_speed
from t
