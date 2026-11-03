-- DP input (NOT itself safe to export). Privacy unit: one node, one cell
-- (sensitivity = clip = dp_count_clip). Grid: the 6 PUBLIC triangle-count
-- bins.
with cells as (
    select * from (values
        ('tri_0', 1), ('tri_1_10', 2), ('tri_11_100', 3),
        ('tri_101_1000', 4), ('tri_1001_10000', 5), ('tri_10001p', 6)
    ) as t(cell, ord)
),

nodes as (
    select node_id, tri_class as cell
    from {{ ref('feat_nodes') }}
)

select
    c.cell,
    coalesce(count(n.node_id), 0)::bigint as clipped_count,
    count(n.node_id)                      as n_contributing_nodes,
    {{ var('dp_count_clip') }}            as clip
from cells c
left join nodes n on n.cell = c.cell
group by c.cell, c.ord
order by c.ord
