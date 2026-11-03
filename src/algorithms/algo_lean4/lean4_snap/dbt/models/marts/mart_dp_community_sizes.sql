-- DP input (NOT itself safe to export). Grid: the 6 PUBLIC community-size
-- bins, NODE-WEIGHTED: one event per NODE, in the bin of ITS community's
-- size. Under the declared row-level privacy unit (one derived node row),
-- adding/removing a row changes exactly one cell by 1 -> sensitivity
-- dp_csize_clip = 1, and the counts are node-scale (thousands), so the
-- release survives the noise. (The community-count histogram — 15 events of
-- sensitivity 2 — was pure noise at any affordable ε: the stock-migration
-- small-n lesson, re-learned here on the second release draw. distill.py
-- converts back to community counts via the PUBLIC bin midpoints — declared
-- post-processing.)
with cells as (
    select * from (values
        ('csize_lt25', 1), ('csize_25_50', 2), ('csize_51_100', 3),
        ('csize_101_200', 4), ('csize_201_400', 5), ('csize_gt400', 6)
    ) as t(cell, ord)
),

node_events as (
    select f.node_id, m.size_bin as cell
    from {{ ref('feat_nodes') }} f
    join {{ ref('mart_community_sizes') }} m
      on m.community_id = f.community_id
)

select
    c.cell,
    coalesce(count(e.node_id), 0)::bigint as clipped_count,
    count(e.node_id)                      as n_contributing_nodes,
    {{ var('dp_csize_clip') }}            as clip
from cells c
left join node_events e on e.cell = c.cell
group by c.cell, c.ord
order by c.ord
