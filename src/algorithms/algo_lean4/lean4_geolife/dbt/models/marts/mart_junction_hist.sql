-- Junction-angle histogram over the public abs-angle classes. Category-2
-- release input (suppression only, declared non-DP). The generator draws
-- its junction turns from these weights (with in-class jitter, so the
-- rendered angles are road-LIKE without sitting on a 90-degree lattice).
with classes as (
    select * from (values
        ('ja_30_60', 1), ('ja_60_120', 2), ('ja_120_180', 3)
    ) as t(ja_class, ord)
)

select
    c.ja_class,
    coalesce(count(s.junction_abs_deg), 0)::bigint as n_junctions,
    count(distinct s.user_id)                      as n_users
from classes c
left join (
    select * from {{ ref('feat_route_segments') }} where mode_class != 'XX'
) s using (ja_class)
group by c.ja_class, c.ord
order by c.ord
