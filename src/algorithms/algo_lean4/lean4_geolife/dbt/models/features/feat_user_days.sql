-- User-days: one row per (user, observed day), with the daily-routine motif
-- class. Classes are mutually exclusive with the public priority order
-- tour > commute > errand_loop > home_day > irregular (a planted tour must
-- classify as a tour even on a commuting day). Unobserved days do NOT
-- appear: recording is bursty, and a day without points is unobserved, not
-- a home day (censoring guideline 5).
with chains as (
    -- distinct non-home anchors reachable inside one day chain
    select
        user_id,
        day_idx,
        day_chain_id,
        count(distinct case when role != 'home' then anchor_id end) as n_nonhome_anchors
    from {{ ref('feat_visits') }}
    group by user_id, day_idx, day_chain_id
),

tourish as (
    select user_id, day_idx, max(n_nonhome_anchors) as max_chain_nonhome
    from chains
    group by user_id, day_idx
),

errands as (
    -- an errand mid-stop: other-anchor visit under the dwell cap, reached
    -- from home and followed by a sequence-ok same-day return home (the
    -- Lean mirror checks the same adjacent pair inside the day list)
    select user_id, day_idx, bool_or(
        role = 'other'
        and dwell_s < {{ var('errand_max_dwell_s') }}
        and trans_cell = 'home__other'
        and next_trans_cell = 'other__home'
        and next_day_idx = day_idx
    ) as has_errand
    from (
        select
            *,
            lead(trans_cell) over w as next_trans_cell,
            lead(day_idx) over w    as next_day_idx
        from {{ ref('feat_visits') }}
        window w as (partition by user_id order by ts_arrive_s, sp_id)
    )
    group by user_id, day_idx
),

d as (
    select
        user_id,
        day_idx,
        count(*)                                           as n_staypoints,
        count(distinct anchor_id)                          as n_anchors,
        bool_or(trans_cell = 'home__work')                 as has_hw,
        bool_or(trans_cell = 'work__home')                 as has_wh,
        bool_and(role = 'home')                            as all_home,
        bool_or(role = 'work')                             as any_work,
        bool_or(is_longdwell)                              as has_longdwell,
        sum(dwell_s)::bigint                               as total_dwell_s,
        any_value(is_weekend)                              as is_weekend
    from {{ ref('feat_visits') }}
    group by user_id, day_idx
)

select
    d.user_id,
    d.day_idx,
    to_timestamp(d.day_idx * 86400)::date                  as day,
    d.n_staypoints,
    d.n_anchors,
    d.total_dwell_s,
    d.has_longdwell,
    d.is_weekend,
    case when d.is_weekend then 'weekend' else 'weekday' end as day_type,
    t.max_chain_nonhome,
    case
        when t.max_chain_nonhome >= {{ var('tour_min_anchors') }} then 'tour'
        when d.has_hw and d.has_wh                                then 'commute'
        when coalesce(e.has_errand, false)                        then 'errand_loop'
        when d.all_home                                           then 'home_day'
        else 'irregular'
    end                                                    as motif_class,
    case
        when d.n_staypoints < {{ var('spdbin_e2') }} then 'spd_1'
        when d.n_staypoints < {{ var('spdbin_e3') }} then 'spd_2'
        when d.n_staypoints < {{ var('spdbin_e4') }} then 'spd_3'
        when d.n_staypoints < {{ var('spdbin_e6') }} then 'spd_4_5'
        else 'spd_6p'
    end                                                    as spd_bin
from d
join tourish t using (user_id, day_idx)
left join errands e using (user_id, day_idx)
