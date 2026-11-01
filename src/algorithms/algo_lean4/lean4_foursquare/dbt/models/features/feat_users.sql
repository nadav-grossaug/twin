-- Per-user profile: travel volume on the public tc bins, the consecutive-pair
-- return rate and its public band, country breadth and gap moments.
-- ENCLAVE-ONLY (per-user rows are identifying); DP releases go through
-- mart_dp_* instead.
with sides as (
    select userid, prev_country as country from {{ ref('feat_transfers') }}
    union all
    select userid, next_country from {{ ref('feat_transfers') }}
),

breadth as (
    select userid, count(distinct country) as n_countries
    from sides
    group by userid
),

u as (
    select
        userid,
        count(*)                                    as n_travels,
        count(*) - 1                                as n_pairs,     -- consecutive pairs
        sum(case when is_return then 1 else 0 end)  as n_returns,
        min(ts_s)                                   as first_ts_s,
        max(ts_s)                                   as last_ts_s,
        avg(gap_s) / 86400.0                        as mean_gap_days,
        stddev_samp(gap_s) / 86400.0                as sd_gap_days
    from {{ ref('feat_transfers') }}
    group by userid
)

select
    u.userid,
    u.n_travels,
    case
        when u.n_travels < {{ var('tc_e2') }}  then 'tc_1'
        when u.n_travels < {{ var('tc_e3') }}  then 'tc_2'
        when u.n_travels < {{ var('tc_e5') }}  then 'tc_3_4'
        when u.n_travels < {{ var('tc_e9') }}  then 'tc_5_8'
        when u.n_travels < {{ var('tc_e17') }} then 'tc_9_16'
        when u.n_travels < {{ var('tc_e33') }} then 'tc_17_32'
        else 'tc_33p'
    end                                                 as tc_bin,
    u.n_pairs,
    u.n_returns,
    100.0 * u.n_returns / nullif(u.n_pairs, 0)          as return_rate_pct,
    -- band undefined below 2 travels (no pair to return on)
    case
        when u.n_pairs = 0 then null
        when 100.0 * u.n_returns / u.n_pairs <     {{ var('rr_band_width_pct') }} then 'rr_0_19'
        when 100.0 * u.n_returns / u.n_pairs < 2 * {{ var('rr_band_width_pct') }} then 'rr_20_39'
        when 100.0 * u.n_returns / u.n_pairs < 3 * {{ var('rr_band_width_pct') }} then 'rr_40_59'
        when 100.0 * u.n_returns / u.n_pairs < 4 * {{ var('rr_band_width_pct') }} then 'rr_60_79'
        else 'rr_80_100'
    end                                                 as rr_band,
    b.n_countries,
    u.first_ts_s,
    u.last_ts_s,
    u.mean_gap_days,
    u.sd_gap_days
from u
join breadth b using (userid)
