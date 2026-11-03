-- Trip-level aggregate moments per leg-count bucket (ENCLAVE-ONLY,
-- release-DENIED aggregate ledger; nothing user-identifying). The reference
-- numbers the realism audit and the synth-vs-real notebook compare against.
select
    n_legs,
    count(*)                                              as n_trips,
    avg(span_days)                                        as mean_span_days,
    avg(total_nights)                                     as mean_total_nights,
    100.0 * sum(case when has_repeat_city then 1 else 0 end)
        / count(*)                                        as repeat_city_pct,
    100.0 * sum(case when is_tour then 1 else 0 end)
        / count(*)                                        as tour_pct,
    100.0 * sum(case when all_contiguous then 1 else 0 end)
        / count(*)                                        as contiguous_pct
from {{ ref('feat_trips') }}
group by n_legs
order by n_legs
