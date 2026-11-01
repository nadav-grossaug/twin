-- Per-transfer features: the public region mapping and transition cell, the
-- gap / stay-duration / hop-distance class taxonomies, calendar fields, the
-- return-motif flag and sequence bookkeeping. All thresholds are public spec
-- vars; unknown countries map to 'XX' via the region_map seed (left join +
-- coalesce), are quality-flagged upstream, and never reach a DP input.
with t as (
    select
        s.*,
        coalesce(pm.region, 'XX') as prev_region,
        coalesce(nm.region, 'XX') as next_region
    from {{ ref('stg_transfers') }} s
    left join {{ ref('region_map') }} pm on pm.country = s.prev_country
    left join {{ ref('region_map') }} nm on nm.country = s.next_country
),

g as (
    select
        *,
        ts_s - lag(ts_s) over w        as gap_s,          -- NULL on first transfer
        row_number() over w            as travel_idx,     -- 1-based per user by dt
        lead(prev_country) over w      as nxt_prev_country,
        lead(next_country) over w      as nxt_next_country,
        -- haversine between the last venue in prev_country and the first in
        -- next_country (km; 6371 = mean Earth radius, a physical constant)
        2 * 6371.0 * asin(least(1.0, sqrt(
            pow(sin(radians(next_lat - prev_lat) / 2), 2)
            + cos(radians(prev_lat)) * cos(radians(next_lat))
              * pow(sin(radians(next_lon - prev_lon) / 2), 2)
        )))                            as dist_km
    from t
    window w as (partition by userid order by dt, obs_id)
)

select
    * exclude (nxt_prev_country, nxt_next_country),
    prev_region || '__' || next_region                        as trans_cell,
    case
        when gap_s is null then null
        when gap_s < {{ var('gap_hop_max_s') }}       then 'gap_hop'
        when gap_s < {{ var('gap_trip_max_s') }}      then 'gap_trip'
        when gap_s < {{ var('gap_monthly_max_s') }}   then 'gap_monthly'
        when gap_s < {{ var('gap_quarterly_max_s') }} then 'gap_quarterly'
        else 'gap_rare'
    end                                                       as gap_class,
    case
        when next_duration_s < {{ var('dur_day_max_s') }}     then 'dur_day'
        when next_duration_s < {{ var('dur_week_max_s') }}    then 'dur_week'
        when next_duration_s < {{ var('dur_month_max_s') }}   then 'dur_month'
        when next_duration_s < {{ var('dur_quarter_max_s') }} then 'dur_quarter'
        else 'dur_long'
    end                                                       as dur_class,
    case
        when dist_km is null then null
        when dist_km < {{ var('dist_short_max_km') }} then 'dist_short'
        when dist_km < {{ var('dist_mid_max_km') }}   then 'dist_mid'
        when dist_km < {{ var('dist_long_max_km') }}  then 'dist_long'
        else 'dist_ultra'
    end                                                       as dist_class,
    -- calendar fields (public facts)
    (isodow(dt) - 1)::smallint                                as dow,     -- 0=Mon..6=Sun
    month(dt)::smallint                                       as month,
    'm' || lpad(month(dt)::varchar, 2, '0')                   as month_cell,
    -- this transfer (a->b) and the user's NEXT one (c->d) form a return
    -- iff c = b and d = a; false on the user's last transfer
    coalesce(nxt_prev_country = next_country
             and nxt_next_country = prev_country, false)      as is_return,
    -- transition rows only count while the sequence is unbroken; the first
    -- transfer (NULL gap) never counts
    coalesce(gap_s <= {{ var('seqbreak_gap_s') }}, false)     as seq_ok
from g
