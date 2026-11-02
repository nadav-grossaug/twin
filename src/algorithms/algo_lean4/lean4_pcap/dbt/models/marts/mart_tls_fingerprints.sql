-- Client-stack population: one row per ClientHello fingerprint (JA3-style,
-- see feat_tls). The RARE-STACK motif flags fingerprints carried by at most
-- `tls_rare_max_flows` flows — an uncommon TLS implementation inside the
-- enclave is a classic tooling/malware lead. Mirrored formally as
-- EnclaveFlows/Tls.lean `IsRareStack` (count over the flow fingerprint list),
-- re-checked by `lake exe checks`.
with per_fp as (
    select
        fingerprint,
        any_value(fp_string)                         as fp_string,
        any_value(c_version)                         as c_version,
        any_value(n_cs)                              as n_cs,
        any_value(n_ext)                             as n_ext,
        count(*)                                     as n_flows,
        count(distinct sa)                           as n_src_hosts,
        count(distinct da)                           as n_dst_hosts,
        count(distinct scs)                          as n_selected_suites,
        mode(scs)                                    as top_scs,
        count(distinct sni_first)                    as n_sni,
        sum(is_canonical_hs::int)                    as n_canonical_hs,
        sum(n_appdata)                               as total_appdata_records,
        min(time_start)                              as first_seen,
        max(time_start)                              as last_seen
    from {{ ref('feat_tls') }}
    group by fingerprint
)

select
    *,
    n_flows::double / sum(n_flows) over ()           as share,
    (n_flows <= {{ var('tls_rare_max_flows') }})     as is_rare_stack
from per_fp
order by n_flows desc
