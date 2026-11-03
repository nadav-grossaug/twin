-- DP input (NOT itself safe to export). Privacy unit: one node, one cell
-- (sensitivity = clip = dp_count_clip). Grid: the 9 PUBLIC feature-richness
-- bins (categories present per node).
with cells as (
    select * from (values
        ('rich_0', 1), ('rich_1', 2), ('rich_2', 3), ('rich_3', 4),
        ('rich_4', 5), ('rich_5', 6), ('rich_6', 7), ('rich_7', 8),
        ('rich_8p', 9)
    ) as t(cell, ord)
),

nodes as (
    select node_id, richness_class as cell
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
