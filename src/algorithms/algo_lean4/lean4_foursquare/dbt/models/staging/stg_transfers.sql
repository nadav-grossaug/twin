-- Canonical country-transfer observations from the travel TSV.
-- One row per (user, transfer); stable obs_id and content-hash obs_key.
-- Region knowledge lives in the seed region_map; here it only feeds the
-- unknown-region quality flag (either side off the public map -> 'XX').
with raw as (
    select *
    from read_csv(
        '{{ var("transfer_path") }}',
        delim = '\t',
        header = true,
        columns = {
            'userid': 'bigint',
            'dt': 'varchar',
            'prev_country': 'varchar',
            'prev_last_venue_id': 'varchar',
            'prev_last_lat': 'double',
            'prev_last_long': 'double',
            'prev_duration': 'bigint',
            'next_country': 'varchar',
            'next_first_venue_id': 'varchar',
            'next_first_lat': 'double',
            'next_first_long': 'double',
            'next_duration': 'bigint'
        }
    )
),

rows_ok as (
    select
        userid,
        strptime(dt, '%Y-%m-%d %H:%M:%S')    as dt,
        prev_country,
        next_country,
        prev_last_venue_id                   as prev_venue,
        next_first_venue_id                  as next_venue,
        prev_last_lat                        as prev_lat,
        prev_last_long                       as prev_lon,
        next_first_lat                       as next_lat,
        next_first_long                      as next_lon,
        prev_duration                        as prev_duration_s,
        next_duration                        as next_duration_s
    from raw
    where userid is not null
      and dt is not null
      and prev_country is not null
      and next_country is not null
)

select
    row_number() over (order by r.userid, r.dt, r.prev_venue, r.next_venue) as obs_id,
    md5(concat_ws('|', r.userid, r.dt, r.prev_country, r.prev_venue,
                  r.prev_lat, r.prev_lon, r.prev_duration_s,
                  r.next_country, r.next_venue, r.next_lat, r.next_lon,
                  r.next_duration_s))                     as obs_key,
    r.userid,
    r.dt,
    epoch(r.dt)::bigint                                   as ts_s,
    r.prev_country,
    r.next_country,
    r.prev_venue,
    r.next_venue,
    r.prev_lat,
    r.prev_lon,
    r.next_lat,
    r.next_lon,
    r.prev_duration_s,
    r.next_duration_s,
    r.prev_duration_s = 0                                 as qf_zero_prev_duration,
    r.next_duration_s = 0                                 as qf_zero_next_duration,
    (pm.region is null or nm.region is null)              as qf_unknown_region
from rows_ok r
left join {{ ref('region_map') }} pm on pm.country = r.prev_country
left join {{ ref('region_map') }} nm on nm.country = r.next_country
