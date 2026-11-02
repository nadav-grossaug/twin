-- Distance-resampled route points: one kept point per route_resample_m of
-- travelled path within a labelled leg, with headings and turning angles
-- recomputed on the resampled stream. Raw 2 s GPS sampling makes per-step
-- turns noise-dominated; at ~25 m spacing a turn is geometry, not jitter.
-- A junction turn is |turn| >= junction_min_turn_deg.
with cum as (
    select
        user_id, trajectory_id, sequence_uuid, mode_class, ts_s,
        latitude, longitude,
        sum(dist_m) over (
            partition by user_id, trajectory_id, sequence_uuid
            order by ts_s rows unbounded preceding) as cum_m
    from {{ ref('feat_steps') }}
),

bucketed as (
    select
        *,
        row_number() over (
            partition by user_id, trajectory_id, sequence_uuid,
                         floor(cum_m / {{ var('route_resample_m') }})
            order by ts_s) as rk
    from cum
),

kept as (
    select user_id, trajectory_id, sequence_uuid, mode_class, ts_s,
           latitude, longitude, cum_m
    from bucketed
    where rk = 1
),

hdg as (
    select
        *,
        cum_m - lag(cum_m) over w as seg_m,
        degrees(atan2(
            (longitude - lag(longitude) over w) * cos(radians(latitude)),
            latitude - lag(latitude) over w)) as heading_deg
    from kept
    window w as (partition by user_id, trajectory_id, sequence_uuid
                 order by ts_s)
)

select
    user_id,
    trajectory_id,
    sequence_uuid,
    mode_class,
    ts_s,
    latitude,
    longitude,
    cum_m,
    seg_m,
    heading_deg,
    -- normalized to (-180, 180]
    case when lag(heading_deg) over w is null then null
         else ((heading_deg - lag(heading_deg) over w + 540) % 360) - 180
    end as turn_deg,
    coalesce(abs(((heading_deg - lag(heading_deg) over w + 540) % 360) - 180)
                 >= {{ var('junction_min_turn_deg') }}, false) as is_junction
from hdg
window w as (partition by user_id, trajectory_id, sequence_uuid order by ts_s)
