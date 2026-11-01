-- Drawdown motif: decline of >= drawdown_num/drawdown_den from the running
-- peak. Per-ticker summary; per-day state lives in feat_returns.in_drawdown.
-- Matches `HasDrawdown` in Motifs.lean (evaluated on the exported price list).
select
    ticker,
    count(*)                                   as n_days,
    max(dd_frac)                               as max_dd_frac,
    sum(case when in_drawdown then 1 else 0 end) as n_days_in_drawdown,
    bool_or(in_drawdown)                       as has_drawdown,
    arg_max(d, dd_frac)                        as worst_day,
    list(price order by d)                     as prices
from {{ ref('feat_returns') }}
group by ticker
order by max_dd_frac desc
