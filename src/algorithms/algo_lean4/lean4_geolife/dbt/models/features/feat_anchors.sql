-- Anchors: ~anchor_cell_m grid clusters of a user's staypoints, with roles.
-- home = the cell with the most night dwell (staypoints spanning the sleep
-- hour); work = the cell != home with the most weekday work-hours dwell
-- (>= anchor_work_min_s); everything else is 'other'. Anchor locations are
-- ENCLAVE-ONLY crown jewels; only roles and role-level statistics may feed
-- releases. The grid uses a per-row cos(lat) equirectangular scale — an
-- approximation that is fine at city scale and declared as such.
with sp as (
    select
        *,
        floor(lat_c * 111320.0 / {{ var('anchor_cell_m') }})::bigint  as cell_lat,
        floor(lon_c * 111320.0 * cos(radians(lat_c))
              / {{ var('anchor_cell_m') }})::bigint                   as cell_lon,
        (mid_dow < 5
         and mid_hour >= {{ var('work_start_h') }}
         and mid_hour < {{ var('work_end_h') }})                      as is_worktime
    from {{ ref('feat_staypoints') }}
),

cells as (
    select
        user_id,
        cell_lat,
        cell_lon,
        count(*)                                                      as n_staypoints,
        sum(dwell_s)::bigint                                          as total_dwell_s,
        sum(case when contains_night_hour then dwell_s else 0 end)::bigint as night_dwell_s,
        sum(case when is_worktime then dwell_s else 0 end)::bigint    as work_dwell_s,
        sum(lat_c * dwell_s) / sum(dwell_s)                           as lat_c,
        sum(lon_c * dwell_s) / sum(dwell_s)                           as lon_c,
        count(distinct day_idx)                                       as n_days
    from sp
    group by user_id, cell_lat, cell_lon
),

ranked as (
    select
        *,
        row_number() over (
            partition by user_id
            order by night_dwell_s desc, total_dwell_s desc, cell_lat, cell_lon
        ) as night_rank
    from cells
),

homed as (
    select
        *,
        (night_rank = 1 and night_dwell_s > 0) as is_home
    from ranked
),

work_ranked as (
    select
        *,
        row_number() over (
            partition by user_id
            order by case when is_home then 0 else work_dwell_s end desc,
                     total_dwell_s desc, cell_lat, cell_lon
        ) as work_rank
    from homed
)

select
    user_id,
    row_number() over (partition by user_id
                       order by total_dwell_s desc, cell_lat, cell_lon) as anchor_id,
    cell_lat,
    cell_lon,
    lat_c,
    lon_c,
    n_staypoints,
    n_days,
    total_dwell_s,
    night_dwell_s,
    work_dwell_s,
    case
        when is_home then 'home'
        when work_rank = 1 and not is_home
             and work_dwell_s >= {{ var('anchor_work_min_s') }} then 'work'
        else 'other'
    end as role
from work_ranked
