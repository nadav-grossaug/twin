-- V-recovery motif: a trough >= vrecov_drop below the running peak that
-- recovers to >= vrecov_level of that peak within vrecov_max_days trading
-- days. Matches `IsVRecovery` in Motifs.lean (peak + window form).
with px as (
    select ticker, d, day_idx, price, peak
    from {{ ref('feat_returns') }}
),

troughs as (
    select *
    from px
    where price <= (1.0 - {{ var('vrecov_drop_num') }} / {{ var('vrecov_drop_den') }}.0) * peak
),

recoveries as (
    select
        t.ticker,
        t.d                        as trough_d,
        t.day_idx                  as trough_idx,
        t.price                    as trough_price,
        t.peak,
        min(p2.day_idx)            as recovery_idx
    from troughs t
    join px p2
      on p2.ticker = t.ticker
     and p2.day_idx > t.day_idx
     and p2.day_idx <= t.day_idx + {{ var('vrecov_max_days') }}
     and p2.price >= {{ var('vrecov_level_num') }} / {{ var('vrecov_level_den') }}.0 * t.peak
    group by 1, 2, 3, 4, 5
)

select
    t.ticker,
    t.d                              as trough_d,
    t.day_idx                        as trough_idx,
    t.price                          as trough_price,
    t.peak,
    r.recovery_idx,
    r.recovery_idx is not null       as is_vrecovery,
    r.recovery_idx - t.day_idx       as days_to_recover
from troughs t
left join recoveries r
  on r.ticker = t.ticker and r.trough_idx = t.day_idx
order by t.ticker, t.d
