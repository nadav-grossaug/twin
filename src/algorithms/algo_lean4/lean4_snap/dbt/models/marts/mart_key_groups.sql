-- Affiliation key-groups: the set of people sharing ONE anonymized feature
-- key (same education instance, same employer, same hometown value...) is an
-- affiliation community, orthogonal to the structural Louvain communities.
-- One row per feature key held by >= 2 nodes, with the number of friendship
-- edges INTERNAL to the group and the resulting pairwise density.
-- ENCLAVE-ONLY (key-level groups are person-identifying; policy denied list).
with keys as (
    select
        feature_key,
        any_value(category)      as category,
        count(distinct node_id)  as n_nodes
    from {{ ref('stg_node_features') }}
    group by feature_key
    having count(distinct node_id) >= 2
),

internal as (
    select f1.feature_key, count(*) as n_internal_edges
    from {{ ref('stg_edges') }} e
    join {{ ref('stg_node_features') }} f1 on f1.node_id = e.src
    join {{ ref('stg_node_features') }} f2
      on f2.node_id = e.dst
     and f2.feature_key = f1.feature_key
    group by f1.feature_key
)

select
    k.feature_key,
    k.category,
    k.n_nodes,
    coalesce(i.n_internal_edges, 0)::bigint as n_internal_edges,
    coalesce(i.n_internal_edges, 0)::double
        / (k.n_nodes * (k.n_nodes - 1) / 2)  as density
from keys k
left join internal i on i.feature_key = k.feature_key
order by k.n_nodes desc, k.feature_key
