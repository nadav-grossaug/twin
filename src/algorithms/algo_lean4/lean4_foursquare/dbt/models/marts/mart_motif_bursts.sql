-- Burst motif: burst_min_transfers consecutive transfers inside a
-- burst_window_s span (frantic-travel signal). Sliding windows of exactly
-- burst_min_transfers transfers; candidates up to TWICE the public window
-- are kept so the Lean agreement check exercises both verdicts (the stock
-- repo's half-threshold candidate discipline); export caps live in
-- python/export_lean_data.py, not here. Matches IsBurst in Motifs.lean.
with w as (
    select
        userid,
        travel_idx                as start_idx,
        count(*) over win         as n_transfers,
        list(ts_s) over win       as ts_list_s,
        max(ts_s) over win - ts_s as window_s
    from {{ ref('feat_transfers') }}
    window win as (partition by userid order by travel_idx
                   rows between current row and {{ var('burst_min_transfers') - 1 }} following)
)

select
    userid,
    start_idx,
    ts_list_s,
    window_s,
    window_s <= {{ var('burst_window_s') }} as is_burst
from w
where n_transfers = {{ var('burst_min_transfers') }}
  and window_s <= {{ var('burst_window_s') }} * 2
order by window_s asc
