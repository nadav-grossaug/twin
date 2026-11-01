-- Corridor motif: an ordered country pair one user repeats with a regular
-- rhythm (the commuter/beacon analog). Gaps are between CONSECUTIVE repeats
-- of the SAME corridor; cv = population sd / mean of those gaps (guarded for
-- mean > 0). All pairs with >= 2 repeats are kept so the Lean agreement
-- check exercises both verdicts; is_corridor applies the public thresholds.
-- Matches IsCorridor in lean/EnclaveTravel/Motifs.lean.
with occ as (
    select
        userid,
        prev_country,
        next_country,
        ts_s,
        ts_s - lag(ts_s) over (
            partition by userid, prev_country, next_country order by ts_s
        ) as rep_gap_s
    from {{ ref('feat_transfers') }}
),

agg as (
    select
        userid,
        prev_country,
        next_country,
        count(*)                                                      as n_repeats,
        list(rep_gap_s order by ts_s) filter (where rep_gap_s is not null) as gaps_s,
        avg(rep_gap_s)                                                as mean_gap_s,
        stddev_pop(rep_gap_s)                                         as sd_gap_s
    from occ
    group by userid, prev_country, next_country
    having count(*) >= 2
),

cvd as (
    select
        *,
        case when mean_gap_s > 0 then sd_gap_s / mean_gap_s end as cv
    from agg
)

select
    userid,
    prev_country,
    next_country,
    n_repeats,
    gaps_s,
    mean_gap_s,
    cv,
    n_repeats >= {{ var('corridor_min_repeats') }}
        and coalesce(cv <= {{ var('corridor_cv_max_num') }}
                           / {{ var('corridor_cv_max_den') }}.0, false) as is_corridor
from cvd
order by n_repeats desc
