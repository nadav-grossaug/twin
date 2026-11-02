-- Segment-length histogram over the public classes (junction-to-junction
-- stretches). Category-2 release input (suppression only, declared non-DP).
with classes as (
    select * from (values
        ('seg_lt50', 1), ('seg_50_150', 2), ('seg_150_600', 3), ('seg_ge600', 4)
    ) as t(seg_class, ord)
)

select
    c.seg_class,
    coalesce(count(s.seg_m), 0)::bigint    as n_segments,
    count(distinct s.user_id)              as n_users
from classes c
left join (
    select * from {{ ref('feat_route_segments') }} where mode_class != 'XX'
) s using (seg_class)
group by c.seg_class, c.ord
order by c.ord
