-- DP input (NOT itself safe to export). Privacy unit: one node, one cell
-- (sensitivity = clip = dp_count_clip). Grid: the 5 PUBLIC
-- circle-membership bins.
with cells as (
    select * from (values
        ('circ_0', 1), ('circ_1', 2), ('circ_2', 3), ('circ_3', 4),
        ('circ_4p', 5)
    ) as t(cell, ord)
),

nodes as (
    select node_id, circle_class as cell
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
