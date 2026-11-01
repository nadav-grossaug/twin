-- Temporal sequence logic I: first-order return-class transitions.
-- For each ticker's date-ordered return sequence (calendar gaps >
-- gap_max_days break the sequence), count class bigrams cur -> next, the
-- conditional probability P(next | cur), and the lift against the marginal
-- P(next). The surge/crash diagonal is the discretized GARCH clustering
-- signal: lift >> 1 on (crash>crash), (surge>surge) means volatility clusters.
with seq as (
    select
        ticker,
        ret_class                     as cur,
        lead(ret_class) over w        as nxt,
        lead(d) over w - d            as gap_days
    from {{ ref('feat_returns') }}
    where ret_class is not null
    window w as (partition by ticker order by d)
),

trans as (
    select cur, nxt, count(*) as n
    from seq
    where nxt is not null and gap_days <= {{ var('gap_max_days') }}
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
    t.n::double / c.n_cur              as p_next,
    (t.n::double / c.n_cur) / m.p_marg as lift
from trans t
join cur_tot c using (cur)
join nxt_marg m using (nxt)
order by t.n desc
