-- Per-leg route-shape profile: junction density, circuity (path / net
-- displacement, the inverse of straightness) with its public class, and
-- segment-length structure. Circuity is defined only for legs that actually
-- went somewhere (net displacement >= route_min_net_m). The LegShapeAE
-- minimodel scores these rows; mart_route_shape aggregates them.
with j as (
    select
        user_id, trajectory_id, sequence_uuid,
        count(*)                   as n_junctions,
        avg(seg_m)                 as mean_seg_m
    from {{ ref('feat_route_segments') }}
    group by user_id, trajectory_id, sequence_uuid
)

select
    l.leg_id,
    l.user_id,
    l.trajectory_id,
    l.sequence_uuid,
    l.mode_class,
    l.path_m,
    l.net_disp_m,
    l.straightness,
    coalesce(j.n_junctions, 0)                           as n_junctions,
    1000.0 * coalesce(j.n_junctions, 0) / nullif(l.path_m, 0) as junctions_per_km,
    j.mean_seg_m,
    case when l.net_disp_m >= {{ var('route_min_net_m') }} and l.net_disp_m > 0
         then l.path_m / l.net_disp_m end                as circuity,
    case when l.net_disp_m >= {{ var('route_min_net_m') }} and l.net_disp_m > 0
         then floor(1000.0 * l.path_m / l.net_disp_m)::bigint end as circ_pm,
    case
        when l.net_disp_m < {{ var('route_min_net_m') }} or l.net_disp_m = 0 then null
        when 1000.0 * l.path_m / l.net_disp_m < {{ var('circ_e1_pm') }} then 'circ_lt120'
        when 1000.0 * l.path_m / l.net_disp_m < {{ var('circ_e2_pm') }} then 'circ_120_150'
        when 1000.0 * l.path_m / l.net_disp_m < {{ var('circ_e3_pm') }} then 'circ_150_250'
        else 'circ_ge250'
    end                                                  as circ_class
from {{ ref('feat_legs') }} l
left join j using (user_id, trajectory_id, sequence_uuid)
