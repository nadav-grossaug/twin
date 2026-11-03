-- Degree-conditioned category presence: the suppression-thresholded release
-- (NO formal guarantee — declared as such in the release policy). Cells
-- with fewer than suppress_min_nodes contributing nodes are nulled and
-- flagged, never emitted with a small count.
with grid as (
    select d.degree_class, c.category
    from (select distinct degree_class from {{ ref('feat_nodes') }}) d
    cross join (select distinct category from {{ ref('category_map') }}) c
),

counts as (
    select
        f.degree_class,
        nf.category,
        count(distinct f.node_id) as n_nodes
    from {{ ref('feat_nodes') }} f
    join {{ ref('stg_node_features') }} nf using (node_id)
    group by f.degree_class, nf.category
),

class_totals as (
    select degree_class, count(*) as class_n
    from {{ ref('feat_nodes') }}
    group by degree_class
)

select
    g.degree_class,
    g.category,
    t.class_n::bigint                                    as class_n,
    case when coalesce(c.n_nodes, 0) >= {{ var('suppress_min_nodes') }}
         then c.n_nodes end::bigint                      as n_nodes,
    coalesce(c.n_nodes, 0) < {{ var('suppress_min_nodes') }} as suppressed
from grid g
join class_totals t using (degree_class)
left join counts c using (degree_class, category)
order by g.degree_class, g.category
