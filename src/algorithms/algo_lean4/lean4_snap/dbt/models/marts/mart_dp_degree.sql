-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one PERSON (node) in SampCert's add/remove-one Neighbour
-- relation — their derived row is one record (row-level DP; the
-- neighbor-induced effect on other rows is a declared assumption, see
-- goal.md). Cell grid: the PUBLIC degree bins, fixed order deg_1..deg_65p.
-- Every node falls in exactly ONE cell: sensitivity = clip = dp_count_clip.
with cells as (
    select * from (values
        ('deg_1', 1), ('deg_2', 2), ('deg_3_4', 3), ('deg_5_8', 4),
        ('deg_9_16', 5), ('deg_17_32', 6), ('deg_33_64', 7), ('deg_65p', 8)
    ) as t(cell, ord)
),

nodes as (
    select node_id, degree_class as cell
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
