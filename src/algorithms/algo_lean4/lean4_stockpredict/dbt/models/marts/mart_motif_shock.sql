-- Shock-day motif: |daily log return| >= shock_bp (earnings-style jumps).
-- Candidates at half the threshold are kept so the Lean agreement check
-- exercises both verdicts. Matches `IsShock` in Motifs.lean.
select
    ticker,
    d,
    day_idx,
    ret_bp,
    ret_class,
    abs(ret_bp) >= {{ var('shock_bp') }} as is_shock
from {{ ref('feat_returns') }}
where ret_bp is not null
  and abs(ret_bp) >= {{ var('shock_bp') }} / 2
order by abs(ret_bp) desc
