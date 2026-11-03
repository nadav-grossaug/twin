-- Canonical node observations from the ingest TSVs.
-- One row per node. nx_* columns are ingest-side cross-checks only:
-- feat_nodes recomputes degree/triangles in SQL and q_data_integrity
-- asserts agreement.
select
    node_id,
    is_ego = 1                                   as is_ego,
    nx_degree::bigint                            as nx_degree,
    nx_triangles::bigint                         as nx_triangles,
    community_id::bigint                         as community_id,
    n_circles::bigint                            as n_circles,
    n_egos::bigint                               as n_egos
from read_csv(
    '{{ var("ingest_dir") }}/nodes.tsv',
    delim = '\t',
    header = true,
    columns = {
        'node_id': 'bigint',
        'is_ego': 'bigint',
        'nx_degree': 'bigint',
        'nx_triangles': 'bigint',
        'community_id': 'bigint',
        'n_circles': 'bigint',
        'n_egos': 'bigint'
    }
)
