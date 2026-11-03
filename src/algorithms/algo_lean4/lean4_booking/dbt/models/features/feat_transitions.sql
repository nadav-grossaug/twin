-- One row per consecutive leg pair within a trip (leg i -> leg i+1).
-- trans_cell is the DP transition mart's cell (prev_ct || '__' || next_ct);
-- gap_cell is the public inter-leg gap class; nights_cell / next_checkin_day
-- describe the arriving (next) leg. The pair's month/dow cells are the
-- arriving leg's calendar fields.
with t as (
    select
        userid,
        utrip,
        leg_idx - 1                  as from_leg,
        prev_city_id,
        city_id                      as next_city_id,
        prev_country,
        hotel_country                as next_country,
        prev_ct,
        hotel_ct                     as next_ct,
        gap_days,
        is_contiguous,
        city_return,
        month_cell,
        dow_cell,
        nights                       as nights_next,
        nights_cell,
        checkin_day                  as next_checkin_day
    from {{ ref('feat_reservations') }}
    where leg_idx >= 2
)

select
    *,
    prev_ct || '__' || next_ct       as trans_cell,
    case
        when gap_days = 0 then 'g_0'
        when gap_days < {{ var('gap_e2') }} then 'g_1'
        when gap_days < {{ var('gap_e4') }} then 'g_2_3'
        else 'g_4p'
    end                             as gap_cell
from t
