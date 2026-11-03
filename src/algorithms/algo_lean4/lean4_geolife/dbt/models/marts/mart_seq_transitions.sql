-- Temporal sequence logic: first-order anchor-role transitions over seq_ok
-- visits (gap <= visit_gap_break_s, same local day; a user's first visit
-- never counts — no false transitions over recording gaps).
-- P(next | prev) and the lift against the marginal P(next): the return-home
-- evening boost is this dataset's analog of the return-trip diagonal.
with trans as (
    select
        prev_role,
        role as next_role,
        count(*) as n
    from {{ ref('feat_visits') }}
    where seq_ok
    group by prev_role, role
),

cur_tot as (
    select prev_role, sum(n) as n_cur from trans group by prev_role
),

nxt_marg as (
    select next_role, sum(n)::double / (select sum(n) from trans) as p_marg
    from trans group by next_role
)

select
    t.prev_role,
    t.next_role,
    t.n,
    t.n::double / c.n_cur              as p_next,
    (t.n::double / c.n_cur) / m.p_marg as lift
from trans t
join cur_tot c using (prev_role)
join nxt_marg m using (next_role)
order by t.n desc
