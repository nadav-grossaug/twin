-- Canonical Cisco Joy flow records.
-- The first line of flow.jsonl is a joy config header (no `sa`); it is filtered out.
with raw as (
    select *
    from read_json_auto(
        '{{ var("flow_path") }}',
        format = 'newline_delimited',
        union_by_name = true,
        sample_size = -1,
        maximum_object_size = 16777216
    )
),

flows as (
    select * from raw
    where sa is not null and time_start is not null
)

select
    row_number() over (order by time_start, sa, da, coalesce(sp, 0), coalesce(dp, 0)) as flow_id,
    md5(concat_ws('|', sa, da, pr, sp, dp, time_start))                              as flow_key,
    sa,
    da,
    pr::integer                                  as pr,
    sp::integer                                  as sp,
    dp::integer                                  as dp,
    coalesce(bytes_out, 0)::bigint               as bytes_out,
    coalesce(bytes_in, 0)::bigint                as bytes_in,
    coalesce(num_pkts_out, 0)::integer           as num_pkts_out,
    coalesce(num_pkts_in, 0)::integer            as num_pkts_in,
    time_start::double                           as time_start,
    time_end::double                             as time_end,
    entropy::double                              as entropy_raw,
    expire_type,
    byte_dist,
    packets,
    dns,
    tls,
    ip['out']['ttl']::integer                    as ttl_out,
    ip['in']['ttl']::integer                     as ttl_in
from flows
