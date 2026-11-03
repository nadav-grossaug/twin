-- Common-knowledge country popularity (suppression thresholds, no formal DP
-- guarantee): countries so widely visited that membership is population-level
-- knowledge, not an individual's trace. ISO codes are a public taxonomy.
-- n_in: transfers arriving into that country (next_country = country).
-- n_out: transfers departing from that country (prev_country = country).
-- n_users: distinct users with at least one arriving or departing transfer.
-- n_transfers: n_in + n_out (total appearances; suppression floor).
with mentions as (
    -- one row per (country, userid) from either direction
    select next_country as country, userid from {{ ref('feat_transfers') }}
    union
    select prev_country as country, userid from {{ ref('feat_transfers') }}
),

user_counts as (
    select country, count(distinct userid) as n_users
    from mentions
    group by country
),

arrivals as (
    select next_country as country, count(*) as n_in
    from {{ ref('feat_transfers') }}
    group by next_country
),

departures as (
    select prev_country as country, count(*) as n_out
    from {{ ref('feat_transfers') }}
    group by prev_country
),

combined as (
    select
        u.country,
        u.n_users,
        coalesce(a.n_in,  0)               as n_in,
        coalesce(d.n_out, 0)               as n_out,
        coalesce(a.n_in,  0)
            + coalesce(d.n_out, 0)         as n_transfers
    from user_counts u
    left join arrivals   a on a.country = u.country
    left join departures d on d.country = u.country
)

select
    c.country,
    coalesce(rm.region, 'XX')  as region,
    c.n_in,
    c.n_out,
    c.n_users,
    c.n_transfers
from combined c
left join {{ ref('region_map') }} rm on rm.country = c.country
where c.n_users     >= {{ var('topk_corridor_min_users') }}
  and c.n_transfers >= {{ var('topk_corridor_min_transfers') }}
order by c.n_transfers desc
