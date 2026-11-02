-- Temporal flow logic I: first-order service transitions.
-- For each host's time-ordered flow sequence (gaps > session_gap_s break the
-- sequence), count service bigrams cur -> next, the conditional probability
-- P(next | cur), and the lift against the marginal P(next): lift >> 1 means
-- "cur is followed by next far more often than chance" — timely flow logic.
with seq as (
    select
        sa,
        service_class as cur,
        lead(service_class) over w as nxt,
        lead(time_start) over w - time_start as gap_s
    from {{ ref('feat_flows') }}
    window w as (partition by sa order by time_start, flow_id)
),

trans as (
    select cur, nxt, count(*) as n
    from seq
    where nxt is not null and gap_s <= {{ var('session_gap_s') }}
    group by cur, nxt
),

cur_tot as (
    select cur, sum(n) as n_cur from trans group by cur
),

nxt_marg as (
    select nxt, sum(n)::double / (select sum(n) from trans) as p_marg
    from trans group by nxt
)

select
    t.cur,
    t.nxt,
    t.n,
    t.n::double / c.n_cur         as p_next,
    (t.n::double / c.n_cur) / m.p_marg as lift
from trans t
join cur_tot c using (cur)
join nxt_marg m using (nxt)
order by t.n desc
