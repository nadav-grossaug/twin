-- Route segments: one row per junction-to-junction stretch of a leg (the
-- distance travelled between two consecutive junction turns), with the
-- public segment-length class and the class of the junction angle that ENDS
-- the segment. This is the "road-like" shape vocabulary the generator
-- renders from: segment lengths + junction angles, both over public bins.
with junctions as (
    select
        user_id, trajectory_id, sequence_uuid, mode_class,
        ts_s, cum_m, turn_deg,
        lag(cum_m) over w as prev_junction_m,
        row_number() over w as junction_idx
    from {{ ref('feat_route_points') }}
    where is_junction
    window w as (partition by user_id, trajectory_id, sequence_uuid
                 order by ts_s)
)

select
    user_id,
    trajectory_id,
    sequence_uuid,
    mode_class,
    junction_idx,
    ts_s,
    (cum_m - prev_junction_m)                           as seg_m,
    case
        when cum_m - prev_junction_m < {{ var('seg_e1_m') }} then 'seg_lt50'
        when cum_m - prev_junction_m < {{ var('seg_e2_m') }} then 'seg_50_150'
        when cum_m - prev_junction_m < {{ var('seg_e3_m') }} then 'seg_150_600'
        else 'seg_ge600'
    end                                                 as seg_class,
    turn_deg                                            as junction_turn_deg,
    abs(turn_deg)                                       as junction_abs_deg,
    case
        when abs(turn_deg) < {{ var('ja_e1_deg') }} then 'ja_30_60'
        when abs(turn_deg) < {{ var('ja_e2_deg') }} then 'ja_60_120'
        else 'ja_120_180'
    end                                                 as ja_class,
    turn_deg > 0                                        as turn_left
from junctions
where prev_junction_m is not null
