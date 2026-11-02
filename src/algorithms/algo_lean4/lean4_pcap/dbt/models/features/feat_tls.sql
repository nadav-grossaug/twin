-- Per-flow TLS features: ClientHello fingerprint (JA3-style: version +
-- offered cipher suites + extension names) and the TLS RECORD STRUCTURE from
-- joy's srlt — type/direction/size/timing of every TLS record, i.e. the
-- packet-level shape of the session that survives encryption.
--
-- The canonical-handshake predicate here is mirrored formally as
-- EnclaveFlows/Tls.lean `IsCanonicalHandshake` and re-checked per flow by
-- `lake exe checks`: the first record is a client-sent handshake record and
-- no application-data record precedes a ChangeCipherSpec in its direction.
--
-- joy version coding: 2 = SSLv3, 3 = TLS 1.0 (this capture is from 2008).
with tls as (
    select
        flow_id,
        sa,
        da,
        dp,
        time_start,
        json_extract(tls, '$.c_version')::integer                 as c_version,
        json_extract(tls, '$.s_version')::integer                 as s_version,
        json_extract_string(tls, '$.scs')                          as scs,
        coalesce(from_json(json_extract(tls, '$.cs'), '["VARCHAR"]'),
                 [])                                               as cs,
        coalesce(from_json(json_extract(tls, '$.sni'), '["VARCHAR"]'),
                 [])                                               as sni,
        coalesce(list_sort(flatten(list_transform(
            from_json(json_extract(tls, '$.c_extensions'), '["JSON"]'),
            e -> json_keys(e)))), [])                              as ext_names,
        (json_extract(tls, '$.c_key_length') is not null)          as has_client_key,
        (json_extract(tls, '$.s_cert') is not null)                as has_server_cert,
        coalesce(from_json(json_extract(tls, '$.srlt'),
            '[{"b":"BIGINT","dir":"VARCHAR","ipt":"BIGINT","tp":"INTEGER"}]'),
            [])                                                    as recs
    from {{ ref('stg_flows') }}
    where tls is not null
),

fp as (
    select
        *,
        -- JA3-style client-stack fingerprint over the hello contents only.
        -- NB: array_to_string([]) is NULL in DuckDB (same NULL-propagation
        -- family as the least() gotcha) — coalesce keeps empty lists as ''.
        'v' || coalesce(c_version::varchar, '?')
            || ',' || coalesce(array_to_string(cs, '-'), '')
            || ',' || coalesce(array_to_string(ext_names, '-'), '') as fp_string,
        -- 1-based first index per (direction, record type); NULL = absent
        list_position(list_transform(recs, r -> r.dir = '>' and r.tp = 20), true) as ccs_out_idx,
        list_position(list_transform(recs, r -> r.dir = '>' and r.tp = 23), true) as data_out_idx,
        list_position(list_transform(recs, r -> r.dir = '<' and r.tp = 20), true) as ccs_in_idx,
        list_position(list_transform(recs, r -> r.dir = '<' and r.tp = 23), true) as data_in_idx
    from tls
)

select
    flow_id,
    sa,
    da,
    dp,
    time_start,
    c_version,
    s_version,
    scs,
    cs,
    len(cs)                                                        as n_cs,
    sni[1]                                                         as sni_first,
    (len(sni) > 0)                                                 as has_sni,
    ext_names,
    len(ext_names)                                                 as n_ext,
    has_client_key,
    has_server_cert,
    fp_string,
    substr(md5(fp_string), 1, 8)                                   as fingerprint,
    recs,
    len(recs)                                                      as n_records,
    len(list_filter(recs, r -> r.tp = 22))                         as n_handshake,
    len(list_filter(recs, r -> r.tp = 23))                         as n_appdata,
    len(list_filter(recs, r -> r.tp = 20))                         as n_ccs,
    len(list_filter(recs, r -> r.tp = 21))                         as n_alert,
    coalesce(list_sum(list_transform(
        list_filter(recs, r -> r.dir = '>'), r -> r.b)), 0)        as tls_bytes_out,
    coalesce(list_sum(list_transform(
        list_filter(recs, r -> r.dir = '<'), r -> r.b)), 0)        as tls_bytes_in,
    list_max(list_transform(recs, r -> r.b))                       as max_record_b,
    -- time spent before the first application-data record (handshake latency)
    coalesce(list_sum(list_transform(
        recs[1:coalesce(least(data_out_idx, data_in_idx),
                        data_out_idx, data_in_idx, len(recs)) - 1],
        r -> r.ipt)), 0)                                           as hs_setup_ms,
    -- shape signature: direction+type tokens of the first 8 records
    array_to_string(list_transform(recs[1:8], r -> r.dir || r.tp), ' ') as shape_sig,
    (len(recs) > 0 and recs[1].dir = '>' and recs[1].tp = 22)      as starts_client_hello,
    -- canonical handshake (mirrors EnclaveFlows/Tls.lean IsCanonicalHandshake)
    (len(recs) > 0 and recs[1].dir = '>' and recs[1].tp = 22
        and (data_out_idx is null or (ccs_out_idx is not null and ccs_out_idx < data_out_idx))
        and (data_in_idx  is null or (ccs_in_idx  is not null and ccs_in_idx  < data_in_idx))
    )                                                              as is_canonical_hs
from fp
