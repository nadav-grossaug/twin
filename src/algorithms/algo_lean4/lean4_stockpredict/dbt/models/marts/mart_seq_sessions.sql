-- Temporal sequence logic II: maximal contiguous trading runs per ticker.
-- A session is a maximal run of trading days where consecutive calendar gaps
-- are <= gap_max_days — the SQL counterpart of the Lean predicate
-- `IsTimelySession` (over day numbers), re-verified by `lake exe checks`.
with seq as (
    select
        ticker,
        d,
        day_num,
        case
            when day_num - lag(day_num) over w > {{ var('gap_max_days') }}
                 or lag(day_num) over w is null
            then 1 else 0
        end as session_start
    from {{ ref('feat_returns') }}
    window w as (partition by ticker order by d)
),

sessions as (
    select
        *,
        sum(session_start) over (partition by ticker order by d) as session_id
    from seq
)

select
    ticker,
    session_id,
    count(*)                                as n_days,
    min(d)                                  as session_start_d,
    max(d)                                  as session_end_d,
    list(day_num order by d)                as day_nums
from sessions
group by ticker, session_id
order by n_days desc
