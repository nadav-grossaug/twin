-- Inter-transfer gap moments per public gap class (days). Suppression to
-- n_gaps >= 200 happens in python/release.py, not here — this mart is the
-- enclave-side ledger the suppressed artifact is derived from.
select
    gap_class,
    count(*)                     as n_gaps,
    avg(gap_s) / 86400.0         as mean_days,
    median(gap_s) / 86400.0      as median_days,
    stddev_samp(gap_s) / 86400.0 as sd_days
from {{ ref('feat_transfers') }}
where gap_s is not null
group by gap_class
order by min(gap_s)
