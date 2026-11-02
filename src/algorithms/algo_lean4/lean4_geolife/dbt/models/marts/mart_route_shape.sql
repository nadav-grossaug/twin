-- Per-mode route-shape summary: circuity moments, pooled junction density
-- and segment structure. The enclave ledger behind the category-2 route
-- release (suppression only, declared non-DP) — the generator renders trip
-- polylines from these numbers plus the seg/junction histograms.
with legs as (
    select
        mode_class,
        count(*)                                   as n_legs,
        count(distinct user_id)                    as n_users,
        median(circuity)                           as median_circuity,
        quantile_cont(circuity, 0.9)               as p90_circuity,
        median(straightness)                       as median_straightness
    from {{ ref('feat_route_legs') }}
    where mode_class != 'XX' and circuity is not null
    group by mode_class
),

dens as (
    select
        mode_class,
        1000.0 * sum(n_junctions) / nullif(sum(path_m), 0) as junctions_per_km,
        avg(case when turn_sign then 1.0 else 0.0 end)     as share_left
    from (
        select l.mode_class, l.path_m, l.n_junctions,
               s.turn_left as turn_sign
        from {{ ref('feat_route_legs') }} l
        left join {{ ref('feat_route_segments') }} s
          using (user_id, trajectory_id, sequence_uuid)
        where l.mode_class != 'XX'
    )
    group by mode_class
)

select
    l.mode_class,
    l.n_legs,
    l.n_users,
    l.median_circuity,
    l.p90_circuity,
    l.median_straightness,
    d.junctions_per_km,
    d.share_left
from legs l
join dens d using (mode_class)
order by l.n_legs desc
