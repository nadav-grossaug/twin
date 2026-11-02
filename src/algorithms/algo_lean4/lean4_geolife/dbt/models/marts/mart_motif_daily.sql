-- Daily-routine motif instances: one row per user-day with its class and
-- witness fields. The Lean mirror re-decides the class from the exported
-- visit lists (priority order tour > commute > errand_loop > home_day >
-- irregular). ENCLAVE-ONLY.
select
    user_id,
    day_idx,
    day,
    day_type,
    motif_class,
    n_staypoints,
    n_anchors,
    max_chain_nonhome,
    has_longdwell,
    spd_bin
from {{ ref('feat_user_days') }}
