-- Per-category coverage over the 11 public categories (full grid: zero
-- rows included so every category appears).
with cats as (
    select distinct category from {{ ref('category_map') }}
),

agg as (
    select
        category,
        count(distinct node_id)     as n_nodes,
        count(distinct feature_key) as n_keys,
        count(*)                    as n_set
    from {{ ref('stg_node_features') }}
    group by category
)

select
    c.category,
    coalesce(a.n_nodes, 0)::bigint as n_nodes,
    coalesce(a.n_keys, 0)::bigint  as n_keys,
    coalesce(a.n_set, 0)::bigint   as n_set
from cats c
left join agg a on a.category = c.category
order by n_nodes desc
