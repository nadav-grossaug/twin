-- Inter-leg gap moments per public gap class (days). Suppression to
-- n_gaps >= gap_moment_min_gaps happens in python/release.py, not here —
-- this mart is the enclave-side ledger the suppressed artifact derives from.
select
    gap_cell          as gap_class,
    count(*)          as n_gaps,
    avg(gap_days)     as mean_days,
    median(gap_days)  as median_days,
    stddev_samp(gap_days) as sd_days
from {{ ref('feat_transitions') }}
group by gap_cell
order by min(gap_days)
