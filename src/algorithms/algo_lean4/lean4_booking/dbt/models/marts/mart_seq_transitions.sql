-- Temporal sequence logic: first-order country-tier transitions within
-- trips (every consecutive leg pair counts; a trip's first leg has no pair
-- and never does). P(next | prev) and the lift against the marginal
-- P(next): the same-country diagonal is the booking analog of route
-- persistence (the return motif). EXACT counts — release-DENIED; the DP
-- counterpart is mart_dp_country_trans.
with trans as (
    select
        prev_ct,
        next_ct,
        count(*) as n
    from {{ ref('feat_transitions') }}
    group by prev_ct, next_ct
),

cur_tot as (
    select prev_ct, sum(n) as n_cur from trans group by prev_ct
),

nxt_marg as (
    select next_ct, sum(n)::double / (select sum(n) from trans) as p_marg
    from trans
    group by next_ct
)

select
    t.prev_ct,
    t.next_ct,
    t.n,
    t.n::double / c.n_cur              as p_next,
    (t.n::double / c.n_cur) / m.p_marg as lift
from trans t
join cur_tot c using (prev_ct)
join nxt_marg m using (next_ct)
order by t.n desc
