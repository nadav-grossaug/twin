-- Revisit beacon: the same non-home anchor visited on >= revisit_min_days
-- distinct days inside a revisit_window_days window (periodic re-contact).
with visit_days as (
    select distinct user_id, anchor_id, role, day_idx
    from {{ ref('feat_visits') }}
    where role != 'home'
),

windows as (
    select
        a.user_id,
        a.anchor_id,
        a.role,
        a.day_idx                 as window_end_day,
        count(distinct b.day_idx) as days_in_window
    from visit_days a
    join visit_days b
      on b.user_id = a.user_id
     and b.anchor_id = a.anchor_id
     and b.day_idx between a.day_idx - ({{ var('revisit_window_days') }} - 1)
                       and a.day_idx
    group by a.user_id, a.anchor_id, a.role, a.day_idx
)

select
    user_id,
    anchor_id,
    any_value(role)                       as role,
    count(distinct window_end_day)        as n_visit_days,
    max(days_in_window)                   as best_window_days,
    max(days_in_window) >= {{ var('revisit_min_days') }} as is_revisit
from windows
group by user_id, anchor_id
