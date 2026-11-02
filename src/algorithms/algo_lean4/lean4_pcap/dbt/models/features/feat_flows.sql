-- Per-flow engineered features.
-- Data-quality fixes documented in goal.md: joy emits small negative durations
-- on single-packet/RST flows (clamped to 0) and occasionally entropy > 8,
-- the theoretical max for bytes (capped to 8, flagged).
select
    flow_id,
    flow_key,
    sa,
    da,
    pr,
    sp,
    dp,
    bytes_out,
    bytes_in,
    num_pkts_out,
    num_pkts_in,
    time_start,
    time_end,
    greatest(time_end - time_start, 0.0)                         as duration_s,
    (time_end - time_start < 0.0)                                as qf_negative_duration,
    bytes_out + bytes_in                                         as bytes_total,
    num_pkts_out + num_pkts_in                                   as pkts_total,
    (bytes_out = 0 and bytes_in = 0)                             as is_zero_byte,
    (num_pkts_in = 0)                                            as is_unanswered,
    case when bytes_in > 0
         then bytes_out::double / bytes_in end                   as byte_ratio_out_in,
    case when time_end - time_start > 0
         then (bytes_out + bytes_in) / (time_end - time_start)
         end                                                     as bytes_per_s,
    -- NB: DuckDB's least() ignores NULLs, so a bare least(entropy_raw, 8.0)
    -- would turn missing entropy into 8.0 (bug found by the synthetic
    -- round-trip audit). NULL must stay NULL.
    case when entropy_raw is null then null
         else least(entropy_raw, 8.0) end                        as entropy,
    (entropy_raw > 8.0)                                          as qf_invalid_entropy,
    expire_type,
    ttl_out,
    ttl_in,
    (sa like '192.168.%' or sa like '10.%' or sa like '169.254.%'
        or regexp_matches(sa, '^172\.(1[6-9]|2[0-9]|3[01])\.'))  as src_internal,
    (da like '192.168.%' or da like '10.%' or da like '169.254.%'
        or da like '2__.%.%.255' or da = '255.255.255.255'
        or regexp_matches(da, '^172\.(1[6-9]|2[0-9]|3[01])\.')
        or regexp_matches(da, '^(22[4-9]|23[0-9])\.'))           as dst_internal,
    -- public, data-independent service taxonomy (the DP cell grid)
    case
        when pr = 1                                  then 'icmp'
        when pr = 2                                  then 'igmp'
        when dp = 53 or sp = 53                      then 'dns'
        when pr = 17 and dp = 5353                   then 'mdns'
        when pr = 17 and dp = 1900                   then 'ssdp'
        when pr = 17 and dp = 123                    then 'ntp'
        when pr = 17 and dp in (67, 68)              then 'dhcp'
        when pr = 17 and dp = 4500                   then 'ipsec_natt'
        when pr = 6 and dp in (80, 8000, 8080)       then 'http'
        when pr = 6 and dp = 443                     then 'https'
        when pr = 6 and dp = 22                      then 'ssh'
        when pr = 6 and dp = 25                      then 'smtp'
        when pr = 6 and dp in (110, 995, 143, 993)   then 'mail'
        when pr = 6 and dp in (5222, 5223, 5190)     then 'chat'
        when pr = 6 and dp = 3283                    then 'ard'
        when dp >= 49152                             then 'ephemeral'
        else 'other'
    end                                                          as service_class,
    cast(strftime(to_timestamp(time_start), '%H') as integer)    as hour_utc,
    (dns is not null)                                            as has_dns,
    (tls is not null)                                            as has_tls,
    (expire_type = 'a')                                          as is_right_censored,
    '{{ var("spec_version") }}'                                  as spec_version,
    byte_dist
from {{ ref('stg_flows') }}
