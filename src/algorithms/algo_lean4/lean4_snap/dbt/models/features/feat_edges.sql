-- One row per canonical undirected edge with endpoint classes, the
-- community-mixing pair cell and the common-neighbor count (the triadic
-- baseline signal). ENCLAVE-ONLY.
with und as (
    select src as a, dst as b from {{ ref('stg_edges') }}
    union all
    select dst as a, src as b from {{ ref('stg_edges') }}
),

cn as (
    select e.src, e.dst, count(*) as common_neighbors
    from {{ ref('stg_edges') }} e
    join und x on x.a = e.src
    join und y on y.a = e.dst and y.b = x.b
    group by e.src, e.dst
),

ranks as (
    select * from (values
        ('c1', 1), ('c2', 2), ('c3', 3), ('c4', 4), ('c5', 5),
        ('c6', 6), ('c7', 7), ('c8', 8), ('rest', 9)
    ) as t(rank_label, ord)
)

select
    e.src,
    e.dst,
    fs.degree_class                                   as src_class,
    fd.degree_class                                   as dst_class,
    fs.community_id = fd.community_id                 as same_community,
    case when rs.ord <= rd.ord
         then rs.rank_label || '__' || rd.rank_label
         else rd.rank_label || '__' || rs.rank_label
    end                                               as pair_cell,
    coalesce(c.common_neighbors, 0)::bigint           as common_neighbors
from {{ ref('stg_edges') }} e
join {{ ref('feat_nodes') }} fs on fs.node_id = e.src
join {{ ref('feat_nodes') }} fd on fd.node_id = e.dst
join ranks rs on rs.rank_label = fs.community_rank
join ranks rd on rd.rank_label = fd.community_rank
left join cn c on c.src = e.src and c.dst = e.dst
