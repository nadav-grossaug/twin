-- Canonical GPS point observations from the GeoLife TSV.
-- One row per kept (user, trajectory, timestamp) point; stable obs_id and
-- content-hash obs_key. Light verified repair, each step counted in
-- q_data_integrity: out-of-range coordinates dropped, exact duplicate
-- (user, trajectory, timestamp) groups reduced to one row (dup_n keeps the
-- group size), the -777 altitude sentinel kept as a declared convention.
-- Mode-class knowledge lives in the seed mode_map; off-map labels become
-- 'XX' and are quality-flagged (none expected on this snapshot).
with raw as (
    select *
    from read_csv(
        '{{ var("points_path") }}',
        delim = '\t',
        header = true,
        columns = {
            'user_id': 'varchar',
            'sequence_uuid': 'varchar',
            'trajectory_id': 'varchar',
            'datetime': 'timestamp',
            'latitude': 'double',
            'longitude': 'double',
            'altitude_ft': 'double',
            'label_start': 'timestamp',
            'label_end': 'timestamp',
            'mode': 'varchar'
        }
    )
),

rows_ok as (
    select *
    from raw
    where user_id is not null
      and trajectory_id is not null
      and datetime is not null
      and latitude between -90 and 90
      and longitude between -180 and 180
),

dedup as (
    select
        *,
        count(*) over (partition by user_id, trajectory_id, datetime) as dup_n,
        row_number() over (
            partition by user_id, trajectory_id, datetime
            order by latitude, longitude, coalesce(mode, '')
        ) as dup_rk
    from rows_ok
)

select
    row_number() over (order by d.user_id, d.datetime, d.trajectory_id)  as obs_id,
    md5(concat_ws('|', d.user_id, d.trajectory_id, d.datetime,
                  d.latitude, d.longitude, d.altitude_ft, d.mode))       as obs_key,
    d.user_id,
    d.sequence_uuid,
    d.trajectory_id,
    -- source timestamps are UTC; every downstream hour / day / weekday
    -- feature is Beijing local time (spec var source_utc_offset_s)
    d.datetime + to_seconds({{ var('source_utc_offset_s') }})            as datetime,
    epoch(d.datetime)::bigint + {{ var('source_utc_offset_s') }}         as ts_s,
    d.latitude,
    d.longitude,
    d.altitude_ft,
    d.mode                                                               as raw_mode,
    coalesce(m.mode_class, 'XX')                                         as mode_class,
    d.dup_n::integer                                                     as dup_n,
    (d.altitude_ft = -777)                                               as qf_alt_missing,
    (m.mode_class is null)                                               as qf_unknown_mode,
    (epoch(d.datetime) + {{ var('source_utc_offset_s') }} < {{ var('window_start_s') }}
     or epoch(d.datetime) + {{ var('source_utc_offset_s') }} >= {{ var('window_end_s') }})
                                                                         as qf_outside_window
from dedup d
left join {{ ref('mode_map') }} m on m.raw_mode = d.mode
where d.dup_rk = 1
