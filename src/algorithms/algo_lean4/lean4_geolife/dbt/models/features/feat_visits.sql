-- Anchor visits: one row per staypoint, carrying its anchor role and the
-- sequence bookkeeping. Visit sequences break at observation gaps >
-- visit_gap_break_s and at day boundaries — a recording gap never creates a
-- false transition (the "breaks split transitions" lemma pattern). Day
-- chains (gap <= tour_max_gap_s within one day) support the tour motif.
with sp as (
    select
        s.*,
        floor(s.lat_c * 111320.0 / {{ var('anchor_cell_m') }})::bigint as cell_lat,
        floor(s.lon_c * 111320.0 * cos(radians(s.lat_c))
              / {{ var('anchor_cell_m') }})::bigint                    as cell_lon
    from {{ ref('feat_staypoints') }} s
),

v as (
    select
        sp.user_id,
        sp.sp_id,
        a.anchor_id,
        a.role,
        sp.lat_c,
        sp.lon_c,
        sp.ts_arrive_s,
        sp.ts_leave_s,
        sp.dwell_s,
        sp.n_points,
        sp.resid_sd_m,
        sp.resid_mad_m,
        sp.resid_p95_m,
        sp.resid_lag1,
        sp.arr_censored,
        sp.dep_censored,
        sp.day_idx,
        sp.arrive_hour,
        sp.leave_hour
    from sp
    join {{ ref('feat_anchors') }} a
      using (user_id, cell_lat, cell_lon)
),

seq as (
    select
        *,
        lag(role)        over w as prev_role,
        lag(ts_leave_s)  over w as prev_leave_s,
        lag(day_idx)     over w as prev_day_idx,
        lag(lat_c)       over w as prev_lat_c,
        lag(lon_c)       over w as prev_lon_c,
        lead(role)       over w as next_role,
        row_number()     over w as visit_idx
    from v
    window w as (partition by user_id order by ts_arrive_s, sp_id)
),

g as (
    select
        *,
        ts_arrive_s - prev_leave_s as gap_s,
        -- an overnight home stay arrives the previous day, so seq_ok is a
        -- pure gap test; day boundaries break day CHAINS (tours), not
        -- transitions. Negative gaps (overlapping detected + implied stays)
        -- are NOT transitions — the ordering is ambiguous (Lean mirrors
        -- this: VisitStep needs prev.leave <= next.arrive).
        coalesce(ts_arrive_s - prev_leave_s
                   between 0 and {{ var('visit_gap_break_s') }},
                 false) as seq_ok,
        2 * 6371000.0 * asin(least(1.0, sqrt(
            pow(sin(radians(lat_c - prev_lat_c) / 2), 2)
            + cos(radians(prev_lat_c)) * cos(radians(lat_c))
              * pow(sin(radians(lon_c - prev_lon_c) / 2), 2)
        )))                        as trip_disp_m
    from seq
)

select
    * exclude (prev_lat_c, prev_lon_c),
    case when seq_ok then prev_role || '__' || role end          as trans_cell,
    case
        when dwell_s < {{ var('dw_e1_s') }} then 'dw_lt1h'
        when dwell_s < {{ var('dw_e2_s') }} then 'dw_1_3h'
        when dwell_s < {{ var('dw_e3_s') }} then 'dw_3_8h'
        when dwell_s < {{ var('dw_e4_s') }} then 'dw_8_14h'
        else 'dw_ge14h'
    end                                                          as dwell_class,
    case
        when arrive_hour <= {{ var('dep_night_max_h') }}     then 'dep_night'
        when arrive_hour <= {{ var('dep_morning_max_h') }}   then 'dep_morning'
        when arrive_hour <= {{ var('dep_afternoon_max_h') }} then 'dep_afternoon'
        else 'dep_evening'
    end                                                          as arr_bin,
    case
        when leave_hour <= {{ var('dep_night_max_h') }}     then 'dep_night'
        when leave_hour <= {{ var('dep_morning_max_h') }}   then 'dep_morning'
        when leave_hour <= {{ var('dep_afternoon_max_h') }} then 'dep_afternoon'
        else 'dep_evening'
    end                                                          as dep_bin,
    case
        when gap_s is null then null
        when trip_disp_m < {{ var('dist_e1_m') }} then 'dist_lt1k'
        when trip_disp_m < {{ var('dist_e2_m') }} then 'dist_1_3k'
        when trip_disp_m < {{ var('dist_e3_m') }} then 'dist_3_10k'
        else 'dist_ge10k'
    end                                                          as dist_class,
    dwell_s >= {{ var('longdwell_min_s') }}                      as is_longdwell,
    ((day_idx + 3) % 7)::smallint                                as dow,  -- 0=Mon
    ((day_idx + 3) % 7) >= 5                                     as is_weekend,
    -- day chains for the tour motif: a new chain starts on a day boundary,
    -- a negative (overlapping) gap, or a gap beyond tour_max_gap_s
    sum(case when gap_s is null
                  or gap_s < 0
                  or gap_s > {{ var('tour_max_gap_s') }}
                  or day_idx != prev_day_idx then 1 else 0 end)
        over (partition by user_id, day_idx
              order by ts_arrive_s, sp_id rows unbounded preceding) as day_chain_id
from g
