-- Per-mode-class kinematic summaries (speed, sampling cadence, turning,
-- straightness, implausibility). The enclave ledger behind the category-2
-- kinematics release (suppression only, declared non-DP).
with steps as (
    select
        mode_class,
        count(*)                                   as n_points,
        count(distinct user_id)                    as n_users,
        median(speed_mps)                          as median_speed_mps,
        quantile_cont(speed_mps, 0.95)             as p95_speed_mps,
        median(dt_s)                               as median_dt_s,
        median(abs(turn_deg))                      as median_abs_turn_deg,
        avg(case when qf_implausible_speed then 1.0 else 0.0 end) as implausible_rate
    from {{ ref('feat_steps') }}
    where mode_class != 'XX'
    group by mode_class
),

legs as (
    select
        mode_class,
        count(*)               as n_legs,
        median(straightness)   as median_straightness,
        median(path_m)         as median_path_m
    from {{ ref('feat_legs') }}
    where mode_class != 'XX'
    group by mode_class
)

select s.*, l.n_legs, l.median_straightness, l.median_path_m
from steps s
left join legs l using (mode_class)
order by s.n_points desc
