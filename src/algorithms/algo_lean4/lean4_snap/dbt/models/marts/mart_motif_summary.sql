-- Motif occupancy: how many nodes carry each public-threshold motif.
-- ENCLAVE-ONLY in per-node form (feat_nodes); these aggregate counts feed
-- the suppressed summary and the DP motif histogram (mart_dp_motifs).
with m as (
    select 'motif_hub' as motif, 1 as ord, count(*) as n_nodes
    from {{ ref('feat_nodes') }} where is_hub
    union all
    select 'motif_leaf', 2, count(*) from {{ ref('feat_nodes') }} where is_leaf
    union all
    select 'motif_broker', 3, count(*) from {{ ref('feat_nodes') }} where is_broker
    union all
    select 'motif_bridge', 4, count(*) from {{ ref('feat_nodes') }} where is_bridge
    union all
    select 'motif_butterfly', 5, count(*) from {{ ref('feat_nodes') }} where is_butterfly
    union all
    select 'motif_core', 6, count(*) from {{ ref('feat_nodes') }} where is_core
)

select
    motif,
    n_nodes::bigint as n_nodes,
    round(n_nodes / (select count(*)::double from {{ ref('feat_nodes') }}), 4) as share
from m
order by ord
