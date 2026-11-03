-- Global triangle/closure statistics. sum(tri_count) over nodes = 3T;
-- transitivity = 3T / wedges = sum(tri_count) / sum(wedge_count);
-- avg_clustering = mean over nodes of cc_num/cc_den (0 when cc_den = 0) —
-- the number that must reproduce the published ~0.6055.
select
    (sum(tri_count) / 3)::bigint                          as total_triangles,
    sum(wedge_count)::bigint                              as total_wedges,
    round(sum(tri_count) / sum(wedge_count)::double, 4)   as transitivity,
    round(avg(case when cc_den > 0
                   then cc_num / cc_den::double
                   else 0 end), 4)                        as avg_clustering,
    count(*)::bigint                                      as n_nodes
from {{ ref('feat_nodes') }}
