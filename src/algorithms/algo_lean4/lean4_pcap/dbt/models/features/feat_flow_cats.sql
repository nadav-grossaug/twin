-- Long-format categorical view of each flow: one row per (flow, feature).
-- The substrate for the chi-square association analysis (mart_feature_assoc):
-- every feature here is categorical, including motif-derived flags, so any
-- pair of features yields a contingency table.
with beacon_channels as (
    select distinct sa, da, dp from {{ ref('mart_motif_beacon') }} where is_beacon
),

fanout_sources as (
    select sa from {{ ref('mart_motif_fanout') }} where is_fanout
),

return_sources as (
    select distinct sa from {{ ref('mart_motif_return') }}
),

tls_flows as (
    select t.flow_id, t.c_version, t.fingerprint, f.is_rare_stack
    from {{ ref('feat_tls') }} t
    join {{ ref('mart_tls_fingerprints') }} f using (fingerprint)
),

dns_flows as (
    select flow_id, count(*) as n_queries, bool_or(is_service_discovery) as any_sd,
           bool_or(is_reverse_lookup) as any_rev
    from {{ ref('feat_dns') }}
    group by flow_id
),

base as (
    select
        f.flow_id,
        f.service_class,
        case when f.src_internal then 'internal' else 'external' end  as src_loc,
        case when f.dst_internal then 'internal' else 'external' end  as dst_loc,
        case when f.is_zero_byte then 'zero_byte' else 'payload' end  as payload,
        case when f.is_unanswered then 'unanswered' else 'answered' end as answered,
        case
            when f.entropy is null then 'none'
            when f.entropy >= {{ var('entropy_high') }} then 'high'
            else 'low'
        end                                                           as entropy_band,
        'h' || lpad(f.hour_utc::varchar, 2, '0')                      as hour_bucket,
        case when b.sa is not null then 'beacon' else 'non_beacon' end as beacon_channel,
        case when fo.sa is not null then 'fanout_src' else 'normal_src' end as fanout_source,
        case when r.sa is not null then 'return_src' else 'no_return_src' end as return_source,
        case
            when t.flow_id is null then 'no_tls'
            else 'tls_v' || coalesce(t.c_version::varchar, '?')
        end                                                           as tls_version,
        case
            when t.flow_id is null then 'no_tls'
            when t.is_rare_stack then 'rare_stack'
            else 'fp_' || t.fingerprint
        end                                                           as tls_stack,
        case
            when d.flow_id is null then 'no_dns'
            when d.any_sd then 'service_discovery'
            when d.any_rev then 'reverse_lookup'
            when d.n_queries > 1 then 'multi_query'
            else 'single_query'
        end                                                           as dns_query_shape
    from {{ ref('feat_flows') }} f
    left join beacon_channels b on f.sa = b.sa and f.da = b.da and f.dp = b.dp
    left join fanout_sources fo on f.sa = fo.sa
    left join return_sources r on f.sa = r.sa
    left join tls_flows t on f.flow_id = t.flow_id
    left join dns_flows d on f.flow_id = d.flow_id
)

unpivot base
on service_class, src_loc, dst_loc, payload, answered,
   entropy_band, hour_bucket, beacon_channel, fanout_source, return_source,
   tls_version, tls_stack, dns_query_shape
into name feature value value
