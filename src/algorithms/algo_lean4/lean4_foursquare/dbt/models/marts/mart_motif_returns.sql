-- Return motif: each consecutive transfer pair of a user — pair1 a->b at
-- ts1, pair2 c->d at ts2 — with is_return = (c = b and d = a). One row per
-- pair (users with a single transfer contribute none); the SQL counterpart
-- of IsReturn in lean/EnclaveTravel/Motifs.lean. c and d are carried beyond
-- the contract's minimum columns because the motifs.json export needs both
-- transfers' endpoints.
with seq as (
    select
        userid,
        travel_idx                as idx,
        prev_country              as a,
        next_country              as b,
        ts_s                      as ts1_s,
        lead(prev_country) over w as c,
        lead(next_country) over w as d,
        lead(ts_s) over w         as ts2_s,
        is_return
    from {{ ref('feat_transfers') }}
    window w as (partition by userid order by travel_idx)
)

select
    userid,
    idx,
    a,
    b,
    c,
    d,
    ts1_s,
    ts2_s,
    ts2_s - ts1_s as gap_s,
    is_return
from seq
where ts2_s is not null
order by ts2_s desc
