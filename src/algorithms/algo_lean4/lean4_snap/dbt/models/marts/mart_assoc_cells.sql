-- Full contingency cells for the spec'd association grids: observed counts,
-- expected counts under independence, and standardized residuals.
-- Grids (contract wiki/snap-contract.md §3):
--   cc_class x degree_class        (structure: embeddedness vs connectivity)
--   degree_class x richness_class  (connectivity vs profile richness)
--   community_rank x motif         (motif occurrences across communities)
--   cat_<a> x cat_<b>              (2x2 present/absent co-occurrence for all
--                                   55 unordered pairs of the 11 categories)
-- mart_feature_assoc aggregates this to per-pair chi2/Cramér's V; the Lean
-- layer recomputes chi-square exactly over ℚ from these cells.
with nodeclass_pairs as (
    select
        'cc_class' as feat_a, 'degree_class' as feat_b,
        cc_class   as val_a,  degree_class   as val_b,
        count(*)   as observed
    from {{ ref('feat_nodes') }}
    group by 3, 4

    union all

    select
        'degree_class', 'richness_class',
        degree_class, richness_class, count(*)
    from {{ ref('feat_nodes') }}
    group by 3, 4
),

motifs as (
    select node_id, 'motif_hub' as motif from {{ ref('feat_nodes') }} where is_hub
    union all
    select node_id, 'motif_leaf' from {{ ref('feat_nodes') }} where is_leaf
    union all
    select node_id, 'motif_broker' from {{ ref('feat_nodes') }} where is_broker
    union all
    select node_id, 'motif_bridge' from {{ ref('feat_nodes') }} where is_bridge
    union all
    select node_id, 'motif_butterfly' from {{ ref('feat_nodes') }} where is_butterfly
    union all
    select node_id, 'motif_core' from {{ ref('feat_nodes') }} where is_core
),

motif_pairs as (
    select
        'community_rank'  as feat_a, 'motif' as feat_b,
        f.community_rank  as val_a,  m.motif as val_b,
        count(*)          as observed
    from motifs m
    join {{ ref('feat_nodes') }} f using (node_id)
    group by 3, 4
),

presence as (
    select
        n.node_id,
        c.category,
        case when p.node_id is not null then 'present' else 'absent' end as val
    from (select node_id from {{ ref('feat_nodes') }}) n
    cross join (select distinct category from {{ ref('category_map') }}) c
    left join (
        select distinct node_id, category from {{ ref('stg_node_features') }}
    ) p on p.node_id = n.node_id and p.category = c.category
),

cat_pairs as (
    select
        'cat_' || p1.category as feat_a,
        'cat_' || p2.category as feat_b,
        p1.val                as val_a,
        p2.val                as val_b,
        count(*)              as observed
    from presence p1
    join presence p2
      on p2.node_id = p1.node_id and p1.category < p2.category
    group by 1, 2, 3, 4
),

pairs as (
    select * from nodeclass_pairs
    union all
    select * from motif_pairs
    union all
    select * from cat_pairs
),

row_marg as (
    select feat_a, feat_b, val_a, sum(observed) as row_n
    from pairs group by 1, 2, 3
),

col_marg as (
    select feat_a, feat_b, val_b, sum(observed) as col_n
    from pairs group by 1, 2, 3
),

totals as (
    select feat_a, feat_b, sum(observed) as n
    from pairs group by 1, 2
),

grid as (
    select r.feat_a, r.feat_b, r.val_a, c.val_b, r.row_n, c.col_n, t.n
    from row_marg r
    join col_marg c using (feat_a, feat_b)
    join totals t using (feat_a, feat_b)
)

select
    g.feat_a,
    g.feat_b,
    g.val_a,
    g.val_b,
    coalesce(p.observed, 0)                    as observed,
    g.row_n,
    g.col_n,
    g.n,
    g.row_n * g.col_n / g.n::double            as expected,
    (coalesce(p.observed, 0) - g.row_n * g.col_n / g.n::double)
        / sqrt(g.row_n * g.col_n / g.n::double) as std_residual
from grid g
left join pairs p
    using (feat_a, feat_b, val_a, val_b)
