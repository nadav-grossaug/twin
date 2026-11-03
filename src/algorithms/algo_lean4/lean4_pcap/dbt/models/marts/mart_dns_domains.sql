-- DNS domain population: one row per registered domain (last two labels).
-- The SUBDOMAIN-FAN motif flags domains queried under many distinct names —
-- the shape of DNS tunneling / DGA / CDN traffic (here: a search-domain
-- artifact `dlr.com` and reverse-lookup zones top the list). Mirrored
-- formally as EnclaveFlows/Dns.lean `IsSubdomainFan` (distinct-count over
-- the queried-name list), re-checked by `lake exe checks`.
with per_domain as (
    select
        registered_domain,
        count(*)                                     as n_queries,
        count(distinct qname)                        as n_distinct_names,
        count(distinct flow_id)                      as n_flows,
        count(distinct sa)                           as n_src_hosts,
        max(n_labels)                                as max_labels,
        max(qname_len)                               as max_qname_len,
        round(avg(qname_len), 1)                     as avg_qname_len,
        round(avg(digit_frac), 3)                    as avg_digit_frac,
        sum(is_service_discovery::int)               as n_service_discovery,
        sum(is_reverse_lookup::int)                  as n_reverse_lookup,
        sum((not is_wellformed)::int)                as n_malformed,
        min(time_start)                              as first_seen,
        max(time_start)                              as last_seen
    from {{ ref('feat_dns') }}
    group by registered_domain
)

select
    *,
    (n_distinct_names >= {{ var('dns_subfan_min_names') }}) as is_subdomain_fan
from per_domain
order by n_distinct_names desc
