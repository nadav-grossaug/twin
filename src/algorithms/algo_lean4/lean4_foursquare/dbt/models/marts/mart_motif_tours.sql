-- Tour motif: sliding windows of exactly tour_min_hops consecutive transfers
-- (by travel_idx) where every intra-window gap is <= tour_max_gap_s;
-- countries = all prev+next countries in the window. Gap-qualified near-miss
-- windows (too few countries, is_tour = false) are kept so the Lean
-- agreement check exercises both verdicts; export caps live in
-- python/export_lean_data.py, not here. Matches IsTour in Motifs.lean.
with w as (
    select
        userid,
        travel_idx                  as start_idx,
        count(*) over win           as n_hops,
        list(ts_s) over win         as ts_list_s,
        list(prev_country) over win as w_prevs,
        list(next_country) over win as w_nexts,
        max(gap_s) over intra       as max_intra_gap_s
    from {{ ref('feat_transfers') }}
    window
        win as (partition by userid order by travel_idx
                rows between current row and {{ var('tour_min_hops') - 1 }} following),
        intra as (partition by userid order by travel_idx
                  rows between 1 following and {{ var('tour_min_hops') - 1 }} following)
),

qualified as (
    select
        userid,
        start_idx,
        ts_list_s,
        list_distinct(w_prevs || w_nexts) as countries,
        n_hops
    from w
    where n_hops = {{ var('tour_min_hops') }}
      and max_intra_gap_s <= {{ var('tour_max_gap_s') }}
)

select
    userid,
    start_idx,
    ts_list_s,
    countries,
    n_hops,
    len(countries)                                    as n_countries,
    len(countries) >= {{ var('tour_min_countries') }} as is_tour
from qualified
order by len(countries) desc, userid, start_idx
