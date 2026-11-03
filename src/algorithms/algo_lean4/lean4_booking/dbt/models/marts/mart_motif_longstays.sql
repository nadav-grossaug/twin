-- Long-stay motif: reservations with nights >= longstay_min_nights (7) —
-- the home-base / extended-stay signal. A near-miss band (nights >=
-- longstay_near_miss_nights) is included so the Lean agreement check
-- exercises both verdicts; export caps live in python/export_lean_data.py.
-- Matches IsLongStay in lean/EnclaveTrip/Motifs.lean.
select
    userid,
    utrip,
    leg_idx as leg,
    nights,
    nights >= {{ var('longstay_min_nights') }} as is_longstay
from {{ ref('feat_reservations') }}
where nights >= {{ var('longstay_near_miss_nights') }}
order by nights desc
