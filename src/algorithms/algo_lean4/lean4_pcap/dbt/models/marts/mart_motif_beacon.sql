-- Beacon motif: a (sa, da, dp) channel re-contacted many times at a
-- near-constant period. Matches the Lean definition `IsBeacon` in
-- lean/EnclaveFlows/Motifs.lean (n >= minFlows, cv <= maxCv, period >= minPeriod).
-- Deterministic baseline (goal.md guideline 2); thresholds are public vars.
select
    sa,
    da,
    dp,
    pr,
    service_class,
    n_flows,
    first_seen,
    last_seen,
    bytes_total,
    mean_gap_s          as period_s,
    median_gap_s,
    sd_gap_s,
    cv_gap,
    n_flows / (1.0 + cv_gap)  as beacon_score,
    (
        n_flows >= {{ var('beacon_min_flows') }}
        and cv_gap is not null
        and cv_gap <= {{ var('beacon_max_cv') }}
        and mean_gap_s >= {{ var('beacon_min_period_s') }}
    ) as is_beacon
from {{ ref('feat_pairs') }}
where n_flows >= {{ var('beacon_min_flows') }}
order by is_beacon desc, cv_gap asc
