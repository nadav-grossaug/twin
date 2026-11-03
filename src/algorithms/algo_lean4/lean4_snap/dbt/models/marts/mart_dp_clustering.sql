-- DP input (NOT itself safe to export). Privacy unit: one node, one cell
-- (sensitivity = clip = dp_count_clip). Grid: the 7 PUBLIC local-clustering
-- classes.
with cells as (
    select * from (values
        ('cc_zero', 1), ('cc_0_20', 2), ('cc_20_40', 3), ('cc_40_60', 4),
        ('cc_60_80', 5), ('cc_80_99', 6), ('cc_one', 7)
    ) as t(cell, ord)
),

nodes as (
    select node_id, cc_class as cell
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
