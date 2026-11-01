-- Common-knowledge corridor release (suppression thresholds, no formal DP
-- guarantee): ordered country pairs so widely travelled that membership is
-- population-level knowledge, not an individual's trace. Thresholds are
-- public vars; regions come from the seed (unknown -> 'XX'; such a pair
-- could only clear the user floor if an off-map country were hugely
-- popular, and the release layer refuses 'XX' regardless).
with c as (
    select
        prev_country,
        next_country,
        count(distinct userid) as n_users,
        count(*)               as n_transfers
    from {{ ref('feat_transfers') }}
    group by prev_country, next_country
)

select
    c.prev_country,
    c.next_country,
    c.n_users,
    c.n_transfers,
    coalesce(pm.region, 'XX') as prev_region,
    coalesce(nm.region, 'XX') as next_region
from c
left join {{ ref('region_map') }} pm on pm.country = c.prev_country
left join {{ ref('region_map') }} nm on nm.country = c.next_country
where c.n_users >= {{ var('topk_corridor_min_users') }}
  and c.n_transfers >= {{ var('topk_corridor_min_transfers') }}
order by c.n_users desc
