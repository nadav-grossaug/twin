-- Per-user moments (ENCLAVE-ONLY, release-DENIED: identifying rows). DP
-- releases go through mart_dp_*; the release policy lists this relation
-- under `denied`.
with u as (
    select userid, n_res, n_trips, n_cities, rc_bin, booker_country, rr, rr_band
    from {{ ref('feat_users') }}
),

t as (
    select
        userid,
        max(n_legs)                                       as max_trip_legs,
        sum(case when is_tour then 1 else 0 end)          as n_tour_trips,
        sum(case when has_repeat_city then 1 else 0 end)   as n_repeat_trips,
        avg(span_days)                                     as mean_span_days,
        avg(total_nights)                                  as mean_total_nights
    from {{ ref('feat_trips') }}
    group by userid
),

g as (
    select
        userid,
        avg(gap_days)         as mean_gap_days,
        stddev_samp(gap_days) as sd_gap_days
    from {{ ref('feat_transitions') }}
    group by userid
)

select
    u.*,
    t.max_trip_legs,
    t.n_tour_trips,
    t.n_repeat_trips,
    t.mean_span_days,
    t.mean_total_nights,
    g.mean_gap_days,
    g.sd_gap_days
from u
join t using (userid)
left join g using (userid)
