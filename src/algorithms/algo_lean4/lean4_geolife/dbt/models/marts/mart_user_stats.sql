-- Per-user profile: observation volume, anchor structure, daily-motif mix
-- and dwell moments. ENCLAVE-ONLY (per-user rows are identifying); DP
-- releases go through mart_dp_* instead. Also the UserDayAE feature source.
with days as (
    select
        user_id,
        count(*)                                                   as n_days,
        avg(n_staypoints)                                          as mean_sp_per_day,
        sum(case when motif_class = 'commute' then 1 else 0 end)   as n_commute_days,
        sum(case when motif_class = 'home_day' then 1 else 0 end)  as n_home_days,
        sum(case when motif_class = 'tour' then 1 else 0 end)      as n_tour_days,
        sum(case when motif_class = 'errand_loop' then 1 else 0 end) as n_errand_days,
        sum(case when motif_class = 'irregular' then 1 else 0 end) as n_irregular_days
    from {{ ref('feat_user_days') }}
    group by user_id
),

anch as (
    select
        user_id,
        count(*)                                        as n_anchors,
        bool_or(role = 'work')                          as has_work,
        bool_or(role = 'home')                          as has_home
    from {{ ref('feat_anchors') }}
    group by user_id
),

vis as (
    select
        user_id,
        count(*)                       as n_visits,
        median(dwell_s)                as median_dwell_s,
        quantile_cont(dwell_s, 0.95)   as p95_dwell_s,
        min(ts_arrive_s)               as first_ts_s,
        max(ts_leave_s)                as last_ts_s
    from {{ ref('feat_visits') }}
    group by user_id
),

legs as (
    select user_id, count(*) as n_legs
    from {{ ref('feat_legs') }}
    group by user_id
)

select
    d.user_id,
    d.n_days,
    d.mean_sp_per_day,
    d.n_commute_days,
    d.n_home_days,
    d.n_tour_days,
    d.n_errand_days,
    d.n_irregular_days,
    d.n_commute_days::double / d.n_days  as share_commute,
    d.n_home_days::double / d.n_days     as share_home,
    a.n_anchors,
    a.has_work,
    a.has_home,
    v.n_visits,
    v.median_dwell_s,
    v.p95_dwell_s,
    v.first_ts_s,
    v.last_ts_s,
    coalesce(l.n_legs, 0)                as n_legs
from days d
join anch a using (user_id)
join vis v using (user_id)
left join legs l using (user_id)
