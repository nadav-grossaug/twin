-- Input to the DP release of the TLS client-stack mix (NOT itself exportable).
-- Privacy unit: one source host (sa), as in mart_dp_counts. Cell names are
-- PUBLIC rank labels (stack_1..stack_5, other) — the rank->fingerprint map
-- stays in-enclave; only rank-labeled clipped counts reach the mechanism, so
-- no fingerprint hash or cipher list ever leaves. Each host's contribution to
-- each cell is clipped at dp_clip.
with ranked as (
    select
        fingerprint,
        row_number() over (order by n_flows desc, fingerprint) as rnk
    from {{ ref('mart_tls_fingerprints') }}
),

cells as (
    select unnest(['stack_1', 'stack_2', 'stack_3', 'stack_4', 'stack_5',
                   'other']) as cell
),

per_user as (
    select
        t.sa,
        case when r.rnk <= 5 then 'stack_' || r.rnk else 'other' end as cell,
        least(count(*), {{ var('dp_clip') }}) as clipped_n
    from {{ ref('feat_tls') }} t
    join ranked r using (fingerprint)
    group by t.sa, cell
)

select
    c.cell,
    coalesce(sum(p.clipped_n), 0)::bigint  as clipped_count,
    count(p.sa)                            as n_contributing_users,
    {{ var('dp_clip') }}                   as clip
from cells c
left join per_user p on p.cell = c.cell
group by c.cell
order by c.cell
