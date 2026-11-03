-- Corridor motif: an ordered country pair one user repeats with a regular
-- rhythm across their trip legs (the commuter/beacon analog on hotel
-- countries). Gaps are between CONSECUTIVE repeats of the SAME ordered pair
-- (days between the pairs' arriving-leg checkins); cv = population sd /
-- mean of those gaps (guarded for mean > 0). All pairs with >= 2 repeats
-- are kept so the Lean agreement check exercises both verdicts;
-- is_corridor applies the public thresholds (min repeats + CV ceiling).
-- Matches IsCorridor in lean/EnclaveTrip/Motifs.lean.
with occ as (
    select
        userid,
        prev_country,
        next_country,
        next_checkin_day as day,
        next_checkin_day - lag(next_checkin_day) over (
            partition by userid, prev_country, next_country
            order by next_checkin_day
        ) as rep_gap_d
    from {{ ref('feat_transitions') }}
),

agg as (
    select
        userid,
        prev_country,
        next_country,
        count(*)                                            as n_repeats,
        list(rep_gap_d order by day)
            filter (where rep_gap_d is not null)            as gaps_days,
        avg(rep_gap_d)                                      as mean_gap_d,
        stddev_pop(rep_gap_d)                               as sd_gap_d
    from occ
    group by userid, prev_country, next_country
    having count(*) >= 2
),

cvd as (
    select
        *,
        case when mean_gap_d > 0 then sd_gap_d / mean_gap_d end as cv
    from agg
)

select
    userid,
    prev_country,
    next_country,
    n_repeats,
    gaps_days,
    mean_gap_d,
    cv,
    n_repeats >= {{ var('corridor_min_repeats') }}
        and coalesce(cv <= {{ var('corridor_cv_max_num') }}
                           / {{ var('corridor_cv_max_den') }}.0, false) as is_corridor
from cvd
order by n_repeats desc
