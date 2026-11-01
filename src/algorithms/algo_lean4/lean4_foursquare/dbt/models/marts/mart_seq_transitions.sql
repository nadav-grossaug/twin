-- Temporal sequence logic: first-order region transitions over seq_ok rows
-- (gap <= seqbreak_gap_s; a user's first transfer has no gap and never
-- counts). 'XX' never enters transition cells — the same filter the DP
-- transition mart applies; off-map rows are tallied in q_data_integrity.
-- P(next | prev) and the lift against the marginal P(next): the return-home
-- diagonal boost is the travel analog of volatility clustering.
with trans as (
    select
        prev_region,
        next_region,
        count(*) as n
    from {{ ref('feat_transfers') }}
    where seq_ok
      and prev_region != 'XX'
      and next_region != 'XX'
    group by prev_region, next_region
),

cur_tot as (
    select prev_region, sum(n) as n_cur from trans group by prev_region
),

nxt_marg as (
    select next_region, sum(n)::double / (select sum(n) from trans) as p_marg
    from trans group by next_region
)

select
    t.prev_region,
    t.next_region,
    t.n,
    t.n::double / c.n_cur              as p_next,
    (t.n::double / c.n_cur) / m.p_marg as lift
from trans t
join cur_tot c using (prev_region)
join nxt_marg m using (next_region)
order by t.n desc
