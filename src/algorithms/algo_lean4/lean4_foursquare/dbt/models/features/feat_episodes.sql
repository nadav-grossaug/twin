-- Journeys (episodes): consecutive transfers chained while the gap to the
-- previous transfer is <= episode_gap_max_s; a larger (or first-transfer
-- NULL) gap opens a new episode (gaps-and-islands). has_tour marks episodes
-- containing a qualifying tour window — same window logic as
-- mart_motif_tours; a qualifying window can never straddle an episode break
-- because tour_max_gap_s <= episode_gap_max_s, and the same-episode
-- condition is checked explicitly anyway.
with s as (
    select
        userid,
        travel_idx,
        ts_s,
        gap_s,
        prev_country,
        next_country,
        case
            when gap_s is null or gap_s > {{ var('episode_gap_max_s') }}
            then 1 else 0
        end as brk
    from {{ ref('feat_transfers') }}
),

e as (
    select
        *,
        sum(brk) over (partition by userid order by travel_idx) as episode_id
    from s
),

-- sliding tour windows of exactly tour_min_hops consecutive transfers
w as (
    select
        userid,
        episode_id,
        count(*) over win            as n_win,
        max(episode_id) over win     as last_episode_id,
        max(gap_s) over intra        as max_intra_gap_s,
        list(prev_country) over win  as w_prevs,
        list(next_country) over win  as w_nexts
    from e
    window
        win as (partition by userid order by travel_idx
                rows between current row and {{ var('tour_min_hops') - 1 }} following),
        intra as (partition by userid order by travel_idx
                  rows between 1 following and {{ var('tour_min_hops') - 1 }} following)
),

tours as (
    select
        userid,
        episode_id,
        bool_or(
            n_win = {{ var('tour_min_hops') }}
            and last_episode_id = episode_id
            and max_intra_gap_s <= {{ var('tour_max_gap_s') }}
            and len(list_distinct(w_prevs || w_nexts)) >= {{ var('tour_min_countries') }}
        ) as has_tour
    from w
    group by userid, episode_id
)

select
    e.userid,
    e.episode_id,
    count(*)                                      as n_hops,
    -- public length bins 1, 2, 3, 4-5, 6+ (identity bins below ep_e4)
    case
        when count(*) = 1 then 'ep_1'
        when count(*) = 2 then 'ep_2'
        when count(*) < {{ var('ep_e4') }} then 'ep_3'
        when count(*) < {{ var('ep_e6') }} then 'ep_4_5'
        else 'ep_6p'
    end                                           as ep_bin,
    min(e.ts_s)                                   as start_ts_s,
    max(e.ts_s)                                   as end_ts_s,
    list_distinct(list(e.prev_country order by e.travel_idx)
                  || list(e.next_country order by e.travel_idx)) as countries,
    coalesce(bool_or(t.has_tour), false)          as has_tour
from e
left join tours t using (userid, episode_id)
group by e.userid, e.episode_id
