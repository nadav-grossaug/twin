-- Common-knowledge city popularity (suppression thresholds, no formal DP
-- guarantee): hotel cities booked so widely that membership is
-- population-level knowledge, not an individual's trace. City ids are the
-- platform's own anonymized taxonomy; thresholds are public vars.
-- n_users: distinct users with at least one reservation in the city.
-- n_res: total reservations in the city (total appearances; floor).
select
    city_id,
    min(hotel_country)   as hotel_country,
    min(hotel_ct)        as hotel_ct,
    count(distinct userid) as n_users,
    count(*)                as n_res
from {{ ref('feat_reservations') }}
group by city_id
having count(distinct userid) >= {{ var('topk_city_min_users') }}
   and count(*)               >= {{ var('topk_city_min_res') }}
order by n_res desc
