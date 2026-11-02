-- Per-query DNS name structure: one row per query record in joy's `dns`
-- blocks — the DNS packet structure that matters in this export (1,488 query
-- records; responses are almost entirely absent, rcode 0 with empty rr).
--
-- Preserved structure: the qname WIRE SHAPE (labels, RFC 1035 length limits,
-- character makeup) and the query's position inside its flow. RFC
-- well-formedness is mirrored formally as EnclaveFlows/Dns.lean
-- `IsWellFormedQName` and re-decided per distinct name by `lake exe checks`.
with recs as (
    select
        flow_id,
        sa,
        da,
        time_start,
        generate_subscripts(from_json(dns, '["JSON"]'), 1)  as rec_idx,
        unnest(from_json(dns, '["JSON"]'))                  as rec
    from {{ ref('stg_flows') }}
    where dns is not null
),

queries as (
    select
        flow_id,
        sa,
        da,
        time_start,
        rec_idx,
        json_extract_string(rec, '$.qn')                    as qname,
        json_extract(rec, '$.rc')::integer                  as rcode,
        coalesce(json_array_length(json_extract(rec, '$.rr')), 0) as n_answers
    from recs
    where json_extract_string(rec, '$.qn') is not null
),

shaped as (
    select
        *,
        length(qname)                                        as qname_len,
        string_split(qname, '.')                             as labels,
        len(string_split(qname, '.'))                        as n_labels,
        list_max(list_transform(string_split(qname, '.'), l -> length(l)))
                                                             as max_label_len,
        list_min(list_transform(string_split(qname, '.'), l -> length(l)))
                                                             as min_label_len,
        length(regexp_replace(qname, '[^0-9]', '', 'g'))::double
            / greatest(length(qname), 1)                     as digit_frac,
        starts_with(qname, '_')                              as is_service_discovery,
        (qname like '%.in-addr.arpa' or qname like '%.ip6.arpa') as is_reverse_lookup
    from queries
)

select
    *,
    case when n_labels >= 2
         then array_to_string(labels[-2:], '.')
         else qname end                                      as registered_domain,
    -- RFC 1035 shape (mirrors EnclaveFlows/Dns.lean IsWellFormedQName)
    (qname_len <= {{ var('dns_max_name_len') }}
        and n_labels >= 1
        and min_label_len >= 1
        and max_label_len <= {{ var('dns_max_label_len') }}) as is_wellformed
from shaped
