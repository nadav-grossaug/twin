-- Tour motif: a day chain (visit gaps <= tour_max_gap_s, one local day)
-- reaching >= tour_min_anchors distinct non-home anchors. The rarest
-- planted pattern for round-trip recall.
select
    user_id,
    day_idx,
    day_chain_id,
    count(distinct case when role != 'home' then anchor_id end) as n_nonhome_anchors,
    count(*)                                                    as n_visits,
    min(ts_arrive_s)                                            as ts_start_s,
    max(ts_leave_s)                                             as ts_end_s,
    count(distinct case when role != 'home' then anchor_id end)
        >= {{ var('tour_min_anchors') }}                        as is_tour
from {{ ref('feat_visits') }}
group by user_id, day_idx, day_chain_id
