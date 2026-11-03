-- One row per node with every derived structural + feature class the marts,
-- DP inputs and the Lean export consume. ENCLAVE-ONLY (per-node rows are
-- identifying).
--
-- Degree, triangles and wedges are computed IN SQL from the canonical edge
-- list (the ingest nx_* columns are cross-checks, asserted equal in
-- q_data_integrity). Local clustering cc = cc_num/cc_den = 2*tri/(d*(d-1))
-- is classified by exact integer cross-multiplication (cc_den * k vs
-- cc_num * cc_den_spec) — no floats, so the Lean mirror decides the same
-- class from the same integers.
with und as (
    select src as a, dst as b from {{ ref('stg_edges') }}
    union all
    select dst as a, src as b from {{ ref('stg_edges') }}
),

deg as (
    select a as node_id, count(*) as degree
    from und
    group by a
),

-- each triangle a<b<c found exactly once on canonical edges
tris as (
    select e1.src as a, e1.dst as b, e2.dst as c
    from {{ ref('stg_edges') }} e1
    join {{ ref('stg_edges') }} e2 on e2.src = e1.dst
    join {{ ref('stg_edges') }} e3 on e3.src = e1.src and e3.dst = e2.dst
),

corners as (
    select a as node_id from tris
    union all
    select b from tris
    union all
    select c from tris
),

tri as (
    select node_id, count(*)::bigint as tri_count
    from corners
    group by node_id
),

comm_sizes as (
    select community_id, count(*) as csize
    from {{ ref('stg_nodes') }}
    group by community_id
),

comm_rank as (
    select
        community_id,
        row_number() over (order by csize desc, community_id) as rk
    from comm_sizes
),

rich as (
    select node_id, count(distinct category) as richness
    from {{ ref('stg_node_features') }}
    group by node_id
),

circ as (
    select
        node_id,
        sum(per_ego)::bigint                            as n_circles,
        max(per_ego)                                    as max_circles_one_ego
    from (
        select node_id, ego_id, count(*) as per_ego
        from {{ ref('stg_circles') }}
        group by node_id, ego_id
    )
    group by node_id
),

neigh_comm as (
    select u.a as node_id, count(distinct n.community_id) as neighbor_communities
    from und u
    join {{ ref('stg_nodes') }} n on n.node_id = u.b
    group by u.a
),

base as (
    select
        s.node_id,
        s.is_ego,
        s.community_id,
        s.n_egos,
        coalesce(d.degree, 0)::bigint                    as degree,
        coalesce(t.tri_count, 0)::bigint                 as tri_count,
        (coalesce(d.degree, 0) * (coalesce(d.degree, 0) - 1) / 2)::bigint as wedge_count,
        (2 * coalesce(t.tri_count, 0))::bigint           as cc_num,
        (coalesce(d.degree, 0) * (coalesce(d.degree, 0) - 1))::bigint     as cc_den,
        coalesce(r.richness, 0)::bigint                  as richness,
        coalesce(c.n_circles, 0)::bigint                 as n_circles,
        coalesce(c.max_circles_one_ego, 0)::bigint       as max_circles_one_ego,
        coalesce(nc.neighbor_communities, 0)::bigint     as neighbor_communities,
        cr.rk                                            as comm_rk
    from {{ ref('stg_nodes') }} s
    left join deg d        on d.node_id = s.node_id
    left join tri t        on t.node_id = s.node_id
    left join rich r       on r.node_id = s.node_id
    left join circ c       on c.node_id = s.node_id
    left join neigh_comm nc on nc.node_id = s.node_id
    left join comm_rank cr on cr.community_id = s.community_id
),

classed as (
    select
        *,
        case
            when degree < {{ var('deg_e2') }}  then 'deg_1'
            when degree < {{ var('deg_e3') }}  then 'deg_2'
            when degree < {{ var('deg_e5') }}  then 'deg_3_4'
            when degree < {{ var('deg_e9') }}  then 'deg_5_8'
            when degree < {{ var('deg_e17') }} then 'deg_9_16'
            when degree < {{ var('deg_e33') }} then 'deg_17_32'
            when degree < {{ var('deg_e65') }} then 'deg_33_64'
            else 'deg_65p'
        end as degree_class,
        case
            when cc_den = 0 or cc_num = 0                    then 'cc_zero'
            when cc_num >= cc_den                            then 'cc_one'
            when cc_num * {{ var('cc_den') }} < cc_den * 1   then 'cc_0_20'
            when cc_num * {{ var('cc_den') }} < cc_den * 2   then 'cc_20_40'
            when cc_num * {{ var('cc_den') }} < cc_den * 3   then 'cc_40_60'
            when cc_num * {{ var('cc_den') }} < cc_den * 4   then 'cc_60_80'
            else 'cc_80_99'
        end as cc_class,
        case
            when tri_count < {{ var('tri_e1') }}     then 'tri_0'
            when tri_count < {{ var('tri_e11') }}    then 'tri_1_10'
            when tri_count < {{ var('tri_e101') }}   then 'tri_11_100'
            when tri_count < {{ var('tri_e1001') }}  then 'tri_101_1000'
            when tri_count < {{ var('tri_e10001') }} then 'tri_1001_10000'
            else 'tri_10001p'
        end as tri_class,
        case
            when richness < {{ var('rich_e8') }} then 'rich_' || richness
            else 'rich_8p'
        end as richness_class,
        case
            when n_circles < {{ var('circ_e4') }} then 'circ_' || n_circles
            else 'circ_4p'
        end as circle_class,
        case
            when comm_rk <= {{ var('n_community_ranks') }} then 'c' || comm_rk
            else 'rest'
        end as community_rank
    from base
)

select
    node_id,
    degree,
    degree_class,
    tri_count,
    wedge_count,
    cc_num,
    cc_den,
    cc_class,
    tri_class,
    community_id,
    community_rank,
    richness,
    richness_class,
    n_circles,
    circle_class,
    max_circles_one_ego,
    n_egos,
    neighbor_communities,
    degree >= {{ var('hub_degree_min') }}                        as is_hub,
    degree = {{ var('leaf_degree') }}                            as is_leaf,
    (degree >= {{ var('broker_degree_min') }}
     and cc_num * {{ var('broker_cc_max_den') }}
         < cc_den * {{ var('broker_cc_max_num') }})              as is_broker,
    neighbor_communities >= {{ var('bridge_communities_min') }}  as is_bridge,
    max_circles_one_ego >= {{ var('butterfly_circles_min') }}    as is_butterfly,
    (degree >= {{ var('core_degree_min') }}
     and cc_num * {{ var('core_cc_min_den') }}
         >= cc_den * {{ var('core_cc_min_num') }})               as is_core,
    is_ego,
    richness = 0                                                 as qf_zero_features
from classed
order by node_id
