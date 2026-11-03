-- Mode legs: one row per labelled same-mode segment (user, trajectory,
-- sequence_uuid). Path length, net displacement, straightness ratio, speed
-- and turning summaries, duration, and the public mode-class and
-- displacement-class taxonomies. Straightness = net displacement / path
-- length (1 = a perfectly straight leg).
with pts as (
    select
        user_id, trajectory_id, sequence_uuid, mode_class,
        ts_s, latitude, longitude
    from {{ ref('stg_points') }}
),

ends as (
    select
        user_id, trajectory_id, sequence_uuid,
        any_value(mode_class)                            as mode_class,
        count(*)                                         as n_points,
        min(ts_s)                                        as ts_start_s,
        max(ts_s)                                        as ts_end_s,
        arg_min(latitude, ts_s)                          as lat_start,
        arg_min(longitude, ts_s)                         as lon_start,
        arg_max(latitude, ts_s)                          as lat_end,
        arg_max(longitude, ts_s)                         as lon_end
    from pts
    group by user_id, trajectory_id, sequence_uuid
),

kin as (
    select
        user_id, trajectory_id, sequence_uuid,
        sum(dist_m)                                      as path_m,
        median(speed_mps)                                as median_speed_mps,
        quantile_cont(speed_mps, 0.95)                   as p95_speed_mps,
        median(dt_s)                                     as median_dt_s,
        median(abs(turn_deg))                            as median_abs_turn_deg,
        sum(case when qf_implausible_speed then 1 else 0 end) as n_implausible
    from {{ ref('feat_steps') }}
    group by user_id, trajectory_id, sequence_uuid
)

select
    row_number() over (order by e.user_id, e.ts_start_s, e.sequence_uuid) as leg_id,
    e.user_id,
    e.trajectory_id,
    e.sequence_uuid,
    e.mode_class,
    e.n_points,
    e.ts_start_s,
    e.ts_end_s,
    e.ts_end_s - e.ts_start_s                            as duration_s,
    k.path_m,
    2 * 6371000.0 * asin(least(1.0, sqrt(
        pow(sin(radians(e.lat_end - e.lat_start) / 2), 2)
        + cos(radians(e.lat_start)) * cos(radians(e.lat_end))
          * pow(sin(radians(e.lon_end - e.lon_start) / 2), 2)
    )))                                                  as net_disp_m,
    case when k.path_m > 0 then
        least(1.0, 2 * 6371000.0 * asin(least(1.0, sqrt(
            pow(sin(radians(e.lat_end - e.lat_start) / 2), 2)
            + cos(radians(e.lat_start)) * cos(radians(e.lat_end))
              * pow(sin(radians(e.lon_end - e.lon_start) / 2), 2)
        ))) / k.path_m) end                              as straightness,
    k.median_speed_mps,
    k.p95_speed_mps,
    k.median_dt_s,
    k.median_abs_turn_deg,
    k.n_implausible,
    case
        when k.path_m < {{ var('dist_e1_m') }} then 'dist_lt1k'
        when k.path_m < {{ var('dist_e2_m') }} then 'dist_1_3k'
        when k.path_m < {{ var('dist_e3_m') }} then 'dist_3_10k'
        else 'dist_ge10k'
    end                                                  as dist_class,
    hour(to_timestamp(e.ts_start_s))::smallint           as start_hour,
    (isodow(to_timestamp(e.ts_start_s)) - 1)::smallint   as dow,
    isodow(to_timestamp(e.ts_start_s)) >= 6              as is_weekend
from ends e
join kin k using (user_id, trajectory_id, sequence_uuid)
