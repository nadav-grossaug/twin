-- DP input (NOT itself safe to export). Privacy unit: one node; each node
-- sees its incident edges' rank-pair cells (an edge appears in BOTH
-- endpoints' vectors — the shared-edge effect is part of the declared
-- row-level DP semantics, goal.md). Each node's cell vector is allocated
-- PROPORTIONALLY (L1, largest-remainder) over least(degree, dp_clip_total):
-- per-cell sensitivity = dp_clip_total.
-- Cell grid: 45 PUBLIC rank-pair cells '<ra>__<rb>', ra <= rb over
-- c1..c8,rest — row-major upper triangle.
with ranks as (
    select * from (values
        ('c1', 1), ('c2', 2), ('c3', 3), ('c4', 4), ('c5', 5),
        ('c6', 6), ('c7', 7), ('c8', 8), ('rest', 9)
    ) as t(rank_label, ord)
),

cells as (
    select
        a.rank_label || '__' || b.rank_label as cell,
        a.ord * 100 + b.ord                  as ord
    from ranks a
    join ranks b on a.ord <= b.ord
),

events as (
    select src as node_id, pair_cell as cell from {{ ref('feat_edges') }}
    union all
    select dst as node_id, pair_cell as cell from {{ ref('feat_edges') }}
),

counts as (
    select node_id, cell, count(*) as n
    from events
    group by node_id, cell
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
