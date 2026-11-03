-- TLS record-structure statistics: flows grouped by handshake SHAPE — the
-- direction+type token sequence of the first 8 TLS records (the part of the
-- packet structure that encryption does not hide). Shapes with the canonical
-- client-first, CCS-before-data layout are separated from violations; the
-- per-flow predicate lives in feat_tls.is_canonical_hs and is re-decided
-- formally per flow by `lake exe checks` (EnclaveFlows/Tls.lean).
select
    shape_sig,
    count(*)                                  as n_flows,
    count(distinct fingerprint)               as n_fingerprints,
    count(distinct sa)                        as n_src_hosts,
    bool_and(starts_client_hello)             as all_start_client_hello,
    bool_and(is_canonical_hs)                 as all_canonical,
    avg(n_records)                            as avg_records,
    avg(n_appdata)                            as avg_appdata,
    avg(n_alert)                              as avg_alerts,
    avg(tls_bytes_out)                        as avg_tls_bytes_out,
    avg(tls_bytes_in)                         as avg_tls_bytes_in,
    avg(hs_setup_ms)                          as avg_hs_setup_ms,
    quantile_cont(hs_setup_ms, 0.5)           as median_hs_setup_ms
from {{ ref('feat_tls') }}
group by shape_sig
order by n_flows desc
