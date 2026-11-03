-- Data-integrity ledger: one named counter per parsing/exclusion/quality
-- event, across pipeline stages, for an at-a-glance audit trail.
-- Counts only — safe to eyeball, still in-enclave.
-- The *_mismatch counters are the SQL<->networkx cross-check: feat_nodes
-- recomputes degree/triangles from edges.tsv; the ingest nx_* columns must
-- agree exactly (0 mismatches) or the pipeline is wrong somewhere.
select 'nodes_parsed' as counter, count(*)::bigint as n, 'stg_nodes' as stage
from {{ ref('stg_nodes') }}
union all
select 'edges_parsed', count(*)::bigint, 'stg_edges'
from {{ ref('stg_edges') }}
union all
select 'self_loops', sum(case when src = dst then 1 else 0 end)::bigint, 'stg_edges'
from {{ ref('stg_edges') }}
union all
select 'noncanonical_edges', sum(case when src >= dst then 1 else 0 end)::bigint, 'stg_edges'
from {{ ref('stg_edges') }}
union all
select 'dup_edges', coalesce(sum(cnt - 1), 0)::bigint, 'stg_edges'
from (
    select count(*) as cnt
    from {{ ref('stg_edges') }}
    group by src, dst
    having count(*) > 1
)
union all
-- the converter bug class (isolated alters in .feat but not .edges): every
-- feature row must reference a staged node
select 'features_without_node', count(*)::bigint, 'stg_node_features'
from {{ ref('stg_node_features') }} f
left join {{ ref('stg_nodes') }} n using (node_id)
where n.node_id is null
union all
select 'circles_without_node', count(*)::bigint, 'stg_circles'
from {{ ref('stg_circles') }} c
left join {{ ref('stg_nodes') }} n using (node_id)
where n.node_id is null
union all
select 'unknown_raw_categories', count(*)::bigint, 'stg_node_features'
from {{ ref('stg_node_features') }} f
left join {{ ref('category_map') }} m on m.raw_category = f.raw_category
where m.raw_category is null
union all
select 'degree_mismatch_nodes',
       sum(case when f.degree != s.nx_degree then 1 else 0 end)::bigint,
       'feat_nodes'
from {{ ref('feat_nodes') }} f
join {{ ref('stg_nodes') }} s using (node_id)
union all
select 'triangle_mismatch_nodes',
       sum(case when f.tri_count != s.nx_triangles then 1 else 0 end)::bigint,
       'feat_nodes'
from {{ ref('feat_nodes') }} f
join {{ ref('stg_nodes') }} s using (node_id)
union all
select 'circle_count_mismatch_nodes',
       sum(case when f.n_circles != s.n_circles then 1 else 0 end)::bigint,
       'feat_nodes'
from {{ ref('feat_nodes') }} f
join {{ ref('stg_nodes') }} s using (node_id)
union all
select 'zero_feature_nodes', sum(case when qf_zero_features then 1 else 0 end)::bigint, 'feat_nodes'
from {{ ref('feat_nodes') }}
union all
select 'multi_ego_nodes', sum(case when n_egos >= 2 then 1 else 0 end)::bigint, 'feat_nodes'
from {{ ref('feat_nodes') }}
union all
select 'isolated_nodes', sum(case when degree = 0 then 1 else 0 end)::bigint, 'feat_nodes'
from {{ ref('feat_nodes') }}
