-- Common-knowledge city-pair corridors (suppression thresholds, no formal
-- DP guarantee): ordered within-trip city pairs so widely travelled that
-- membership is population-level knowledge, not an individual's trace —
-- the next-destination structure the generator draws cities from when the
-- current city has a released successor pool.
-- n_users: distinct users making the hop; n_trans: total hops (floor).
select
    prev_city_id                          as prev_city,
    next_city_id                          as next_city,
    min(next_ct)                          as hotel_ct,
    count(distinct userid)                as n_users,
    count(*)                              as n_trans
from {{ ref('feat_transitions') }}
group by prev_city_id, next_city_id
having count(distinct userid) >= {{ var('topk_trans_min_users') }}
   and count(*)               >= {{ var('topk_trans_min_trans') }}
order by n_trans desc
