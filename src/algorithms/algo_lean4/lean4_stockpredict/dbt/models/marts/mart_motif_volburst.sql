-- Volatility-burst motif: 5-day rolling sample sd of daily log returns
-- >= volburst_min_bp. Candidates above half the threshold are kept so the
-- Lean agreement check exercises both verdicts. The window's returns are
-- exported for `IsVolBurst` (Motifs.lean), which restates sd >= thr as
-- ssd >= thr^2 * (n-1) over exact rationals.
with w as (
    select
        ticker,
        d,
        day_idx,
        vol5_bp,
        vol5_n,
        is_volburst,
        list(ret_bp) over (
            partition by ticker order by d
            rows between {{ var('volburst_window') - 1 }} preceding and current row
        ) as window_rets_bp
    from {{ ref('feat_returns') }}
    where ret_bp is not null
)

select *
from w
where vol5_n >= {{ var('volburst_window') }}
  and vol5_bp >= {{ var('volburst_min_bp') }} / 2.0
order by vol5_bp desc
