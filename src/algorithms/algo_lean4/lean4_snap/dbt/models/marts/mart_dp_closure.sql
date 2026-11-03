-- DP input (NOT itself safe to export). Privacy unit: one node; its wedge
-- vector — closed = tri_count, open = wedge_count - tri_count, both in the
-- node's own degree-class cells — is allocated PROPORTIONALLY (L1,
-- largest-remainder) over least(wedge_count, dp_clip_total): per-cell
-- sensitivity = dp_clip_total. This grid is the triadic-closure
-- conditional the generator targets.
-- Cell grid: 16 PUBLIC cells 'clo_<degree_class>__closed/open' in
-- degree-class order.
with classes as (
    select * from (values
        ('deg_1', 1), ('deg_2', 2), ('deg_3_4', 3), ('deg_5_8', 4),
        ('deg_9_16', 5), ('deg_17_32', 6), ('deg_33_64', 7), ('deg_65p', 8)
    ) as t(degree_class, ord)
),

cells as (
    select 'clo_' || degree_class || '__closed' as cell, ord * 2 - 1 as ord
    from classes
    union all
    select 'clo_' || degree_class || '__open', ord * 2
    from classes
),

counts as (
    select node_id, 'clo_' || degree_class || '__closed' as cell,
           tri_count as n
    from {{ ref('feat_nodes') }}
    where tri_count > 0
    union all
    select node_id, 'clo_' || degree_class || '__open',
           wedge_count - tri_count
    from {{ ref('feat_nodes') }}
    where wedge_count - tri_count > 0
),

totals as (
    select
        node_id,
        sum(n) as total,
        least(sum(n), {{ var('dp_clip_total') }}) as budget
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
    {{ var('dp_clip_total') }}                                   as clip
from cells c
left join per_node p on p.cell = c.cell
group by c.cell, c.ord
order by c.ord
