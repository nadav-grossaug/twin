-- Run motif: maximal same-sign streaks of daily log returns per ticker
-- (gaps-and-islands). Matches the Lean definition `IsRun` in
-- lean/EnclaveSeries/Motifs.lean (length >= run_min_days, all signs equal and
-- nonzero). Deterministic baseline; thresholds are public vars.
with s as (
    select
        ticker,
        d,
        day_idx,
        ret_bp,
        ret_sign,
        case
            when ret_sign != lag(ret_sign) over w
                 or lag(ret_sign) over w is null
                 or coalesce(gap_days > {{ var('gap_max_days') }}, false)
            then 1 else 0
        end as brk
    from {{ ref('feat_returns') }}
    where ret_bp is not null
    window w as (partition by ticker order by d)
),

grp as (
    select
        *,
        sum(brk) over (partition by ticker order by d) as run_id
    from s
)

select
    ticker,
    run_id,
    any_value(ret_sign)                 as sgn,
    count(*)                            as len_days,
    min(d)                              as start_d,
    max(d)                              as end_d,
    sum(ret_bp)                         as total_bp,
    list(ret_bp order by d)             as rets_bp,
    any_value(ret_sign) != 0
        and count(*) >= {{ var('run_min_days') }} as is_run
from grp
group by ticker, run_id
order by len_days desc
