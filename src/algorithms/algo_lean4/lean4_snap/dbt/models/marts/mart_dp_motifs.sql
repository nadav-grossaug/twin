-- DP input (NOT itself safe to export). Privacy unit: one node; its motif
-- presence vector (<= 1 per motif, up to 6 total) is allocated
-- PROPORTIONALLY (L1, largest-remainder) over least(n_motifs,
-- dp_clip_motifs): per-cell sensitivity = dp_clip_motifs.
-- Cell grid: the 6 PUBLIC motifs, fixed order.
with cells as (
    select * from (values
        ('motif_hub', 1), ('motif_leaf', 2), ('motif_broker', 3),
        ('motif_bridge', 4), ('motif_butterfly', 5), ('motif_core', 6)
    ) as t(cell, ord)
),

counts as (
    select node_id, 'motif_hub' as cell, 1::bigint as n
    from {{ ref('feat_nodes') }} where is_hub
    union all
    select node_id, 'motif_leaf', 1 from {{ ref('feat_nodes') }} where is_leaf
    union all
    select node_id, 'motif_broker', 1 from {{ ref('feat_nodes') }} where is_broker
    union all
    select node_id, 'motif_bridge', 1 from {{ ref('feat_nodes') }} where is_bridge
    union all
    select node_id, 'motif_butterfly', 1 from {{ ref('feat_nodes') }} where is_butterfly
    union all
    select node_id, 'motif_core', 1 from {{ ref('feat_nodes') }} where is_core
),

totals as (
    select
        node_id,
        sum(n) as total,
        least(sum(n), {{ var('dp_clip_motifs') }}) as budget
    from counts
    group by node_id
),

alloc as (
    select
        c.node_id,
        c.cell,
        (c.n * t.budget) // t.total as base,
        (c.n * t.budget) % t.total  as frac,
        t.budget
    from counts c
    join totals t using (node_id)
),

ranked as (
    select
        *,
        row_number() over (partition by node_id order by frac desc, cell) as rk,
        budget - sum(base) over (partition by node_id) as remainder
    from alloc
),

per_node as (
    select
        node_id,
        cell,
        (base + case when rk <= remainder then 1 else 0 end)::bigint as clipped_n
    from ranked
)

select
    c.cell,
    coalesce(sum(p.clipped_n), 0)::bigint                        as clipped_count,
    count(distinct case when p.clipped_n > 0 then p.node_id end) as n_contributing_nodes,
    {{ var('dp_clip_motifs') }}                                  as clip
from cells c
left join per_node p on p.cell = c.cell
group by c.cell, c.ord
order by c.ord
