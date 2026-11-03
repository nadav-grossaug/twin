-- Data-integrity ledger: one named counter per parsing/exclusion/quality
-- event, across pipeline stages, for an at-a-glance audit trail.
-- Counts only — safe to eyeball, still in-enclave.
select 'reservations_parsed' as counter, count(*)::bigint as n, 'stg_reservations' as stage
from {{ ref('stg_reservations') }}
union all
select 'bad_date_rows', sum(case when qf_bad_dates then 1 else 0 end)::bigint, 'stg_reservations'
from {{ ref('stg_reservations') }}
union all
select 'overlap_rows_removed', sum(case when qf_overlap then 1 else 0 end)::bigint, 'stg_reservations'
from {{ ref('stg_reservations') }}
union all
select 'outside_window_rows',
       sum(case when checkin_day < {{ var('window_start_day') }}
                  or checkin_day >= {{ var('window_end_day') }} then 1 else 0 end)::bigint,
       'stg_reservations'
from {{ ref('stg_reservations') }}
union all
select 'dup_utrip_leg_rows', coalesce(sum(cnt - 1), 0)::bigint, 'stg_reservations'
from (
    select count(*) as cnt
    from {{ ref('stg_reservations') }}
    group by utrip, leg_idx
    having count(*) > 1
)
union all
select 'null_field_rows',
       sum(case when userid is null or city_id is null or device_class is null
                     or affiliate_id is null or booker_country is null
                     or hotel_country is null then 1 else 0 end)::bigint,
       'stg_reservations'
from {{ ref('stg_reservations') }}
union all
select 'legs', count(*)::bigint, 'feat_reservations'
from {{ ref('feat_reservations') }}
union all
select 'trips', count(*)::bigint, 'feat_trips'
from {{ ref('feat_trips') }}
union all
select 'users', count(*)::bigint, 'feat_users'
from {{ ref('feat_users') }}
union all
select 'transitions', count(*)::bigint, 'feat_transitions'
from {{ ref('feat_transitions') }}
union all
select 'noncontiguous_transitions', sum(case when not is_contiguous then 1 else 0 end)::bigint, 'feat_transitions'
from {{ ref('feat_transitions') }}
union all
select 'repeat_city_trips', sum(case when has_repeat_city then 1 else 0 end)::bigint, 'feat_trips'
from {{ ref('feat_trips') }}
union all
select 'tour_trips', sum(case when is_tour then 1 else 0 end)::bigint, 'feat_trips'
from {{ ref('feat_trips') }}
union all
select 'longstay_legs', count(*)::bigint, 'mart_motif_longstays'
from {{ ref('mart_motif_longstays') }} where is_longstay
union all
select 'corridor_motifs', count(*)::bigint, 'mart_motif_corridors'
from {{ ref('mart_motif_corridors') }} where is_corridor
