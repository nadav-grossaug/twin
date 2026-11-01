-- Per-user travel statistics: volume, breadth, return behaviour, gap and
-- distance moments, and gap-class occupancy.
-- ENCLAVE-ONLY (per-user rows are identifying); DP releases go through
-- mart_dp_* instead.
with t as (
    select
        userid,
        median(gap_s) / 86400.0                                    as median_gap_days,
        min(gap_s)                                                 as min_gap_s,
        max(gap_s)                                                 as max_gap_s,
        avg(dist_km)                                               as mean_dist_km,
        max(dist_km)                                               as max_dist_km,
        count(distinct next_region)                                as n_regions,
        sum(case when seq_ok then 1 else 0 end)                    as n_seq_ok_pairs,
        sum(case when gap_class = 'gap_hop' then 1 else 0 end)     as n_gap_hop,
        sum(case when gap_class = 'gap_trip' then 1 else 0 end)    as n_gap_trip,
        sum(case when gap_class = 'gap_monthly' then 1 else 0 end) as n_gap_monthly,
        sum(case when gap_class = 'gap_quarterly' then 1 else 0 end) as n_gap_quarterly,
        sum(case when gap_class = 'gap_rare' then 1 else 0 end)    as n_gap_rare
    from {{ ref('feat_transfers') }}
    group by userid
),

e as (
    select
        userid,
        count(*)                                  as n_episodes,
        sum(case when has_tour then 1 else 0 end) as n_tour_episodes,
        max(n_hops)                               as max_episode_hops
    from {{ ref('feat_episodes') }}
    group by userid
)

select
    u.userid,
    u.n_travels,
    u.tc_bin,
    u.n_pairs,
    u.n_returns,
    u.return_rate_pct,
    u.rr_band,
    u.n_countries,
    u.first_ts_s,
    u.last_ts_s,
    (u.last_ts_s - u.first_ts_s) / 86400.0 as span_days,
    u.mean_gap_days,
    u.sd_gap_days,
    t.median_gap_days,
    t.min_gap_s,
    t.max_gap_s,
    t.mean_dist_km,
    t.max_dist_km,
    t.n_regions,
    t.n_seq_ok_pairs,
    t.n_gap_hop,
    t.n_gap_trip,
    t.n_gap_monthly,
    t.n_gap_quarterly,
    t.n_gap_rare,
    e.n_episodes,
    e.n_tour_episodes,
    e.max_episode_hops
from {{ ref('feat_users') }} u
join t using (userid)
join e using (userid)
order by userid
