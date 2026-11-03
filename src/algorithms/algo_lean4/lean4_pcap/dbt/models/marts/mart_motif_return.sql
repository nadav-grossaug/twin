-- Return-trip motif A -> B -> A over each source host's destination sequence
-- (the network analogue of the GPS return-trip motif). Sequences are ordered
-- by time_start per source; a gap larger than session_gap_s breaks the
-- sequence (goal.md guideline 5: respect incomplete observations).
with seq as (
    select
        sa,
        da,
        time_start,
        service_class,
        lag(da, 1)  over w as prev_da,
        lag(da, 2)  over w as prev2_da,
        time_start - lag(time_start, 1) over w as gap1_s,
        time_start - lag(time_start, 2) over w as gap2_s
    from {{ ref('feat_flows') }}
    window w as (partition by sa order by time_start, flow_id)
),

hits as (
    select
        sa,
        prev2_da          as dst_a,
        prev_da           as dst_b,
        time_start        as return_time,
        gap2_s            as roundtrip_s,
        service_class
    from seq
    where da = prev2_da
      and da <> prev_da
      and gap1_s <= {{ var('return_max_gap_s') }}
      and gap2_s <= 2 * {{ var('return_max_gap_s') }}
      -- a gap above session_gap_s would have broken the sequence
      and gap2_s <= {{ var('session_gap_s') }}
)

select
    sa,
    dst_a,
    dst_b,
    count(*)            as n_occurrences,
    min(return_time)    as first_seen,
    max(return_time)    as last_seen,
    avg(roundtrip_s)    as mean_roundtrip_s,
    any_value(service_class) as service_class
from hits
group by sa, dst_a, dst_b
order by n_occurrences desc
