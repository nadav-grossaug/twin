-- DP input (NOT itself safe to export). Privacy unit: one node; the events
-- are its COHESIVE-KEY MEMBERSHIPS — one event per feature key of a released
-- affiliation category that the node holds and that is shared by >= 2 nodes,
-- in the PUBLIC size bin of that key-group (affil_g5/affil_g9/affil_g17
-- edges). The per-node event vector is allocated PROPORTIONALLY (L1,
-- largest-remainder) over a budget of least(n_memberships, dp_affil_clip), so
-- it sums to <= dp_affil_clip and per-cell sensitivity equals the clip — the
-- proved mechanism applies unchanged (mart_dp_categories semantics). The
-- released histogram is MEMBER-WEIGHTED: distill converts to group counts via
-- public midpoints.
-- Cell grid (fixed order): aff_<category>_<size_bin> over the released
-- categories x (g_2_4, g_5_8, g_9_16, g_17p). The category list MUST equal
-- spec.py AFFIL_RELEASE_CATS (tests/test_spec_sync.py checks it).
{% set affil_cats = ['birthday', 'education', 'gender', 'hometown', 'languages', 'last_name', 'locale', 'location', 'work'] %}
{% set affil_bins = ['g_2_4', 'g_5_8', 'g_9_16', 'g_17p'] %}
with cells as (
    select * from (values
    {%- set ns = namespace(i=0) %}
    {%- for c in affil_cats %}{% for b in affil_bins %}{% set ns.i = ns.i + 1 %}
        ('aff_{{ c }}_{{ b }}', {{ ns.i }}){{ "," if ns.i < affil_cats | length * affil_bins | length else "" }}
    {%- endfor %}{% endfor %}
    ) as t(cell, ord)
),

groups as (
    select
        feature_key,
        category,
        case
            when n_nodes < {{ var('affil_g5') }} then 'g_2_4'
            when n_nodes < {{ var('affil_g9') }} then 'g_5_8'
            when n_nodes < {{ var('affil_g17') }} then 'g_9_16'
            else 'g_17p'
        end as size_bin
    from {{ ref('mart_key_groups') }}
    where category in ({% for c in affil_cats %}'{{ c }}'{{ ", " if not loop.last else "" }}{% endfor %})
),

counts as (
    select
        f.node_id,
        'aff_' || g.category || '_' || g.size_bin as cell,
        count(*)::bigint                          as n
    from {{ ref('stg_node_features') }} f
    join groups g on g.feature_key = f.feature_key
    group by 1, 2
),

totals as (
    select
        node_id,
        sum(n) as total,
        least(sum(n), {{ var('dp_affil_clip') }}) as budget
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
    {{ var('dp_affil_clip') }}                                   as clip
from cells c
left join per_node p on p.cell = c.cell
group by c.cell, c.ord
order by c.ord
