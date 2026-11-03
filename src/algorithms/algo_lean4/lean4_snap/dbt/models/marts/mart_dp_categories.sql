-- DP input (NOT itself safe to export). Privacy unit: one node; its
-- category-presence vector (<= 1 per category, up to 11 total) is allocated
-- PROPORTIONALLY (L1, largest-remainder) over a budget of
-- least(n_categories, dp_clip_total), so the vector sums to <= dp_clip_total
-- and per-cell sensitivity equals the clip — the proved mechanism applies
-- unchanged (foursquare mart_dp_region semantics).
-- Cell grid: the 11 PUBLIC categories, contract order.
with cells as (
    select * from (values
        ('cat_birthday', 1), ('cat_education', 2), ('cat_first_name', 3),
        ('cat_gender', 4), ('cat_hometown', 5), ('cat_languages', 6),
        ('cat_last_name', 7), ('cat_locale', 8), ('cat_location', 9),
        ('cat_work', 10), ('cat_other', 11)
    ) as t(cell, ord)
),

counts as (
    select node_id, 'cat_' || category as cell, 1::bigint as n
    from (
        select distinct node_id, category
        from {{ ref('stg_node_features') }}
    )
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
