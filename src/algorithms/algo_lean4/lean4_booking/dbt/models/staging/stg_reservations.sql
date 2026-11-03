-- Canonical multi-destination trip reservations from the booking CSV.
-- One row per (utrip, leg); stable obs_id and content-hash obs_key.
-- Country/affiliate tier knowledge lives in the seeds; here it feeds the
-- hotel_ct / aff_tier columns (unmapped -> 'ct_other' / 'aff_other' — the
-- 15 named hotel-country tiers and 8 affiliate tiers are public constants;
-- everything else pools into the catch-all cell).
with raw as (
    select * from read_csv(
        '{{ var("booking_csv_path") }}',
        header = true,
        columns = {
            'user_id': 'bigint',
            'checkin': 'varchar',
            'checkout': 'varchar',
            'city_id': 'bigint',
            'device_class': 'varchar',
            'affiliate_id': 'bigint',
            'booker_country': 'varchar',
            'hotel_country': 'varchar',
            'utrip_id': 'varchar'
        }
    )
),

rows_ok as (
    select
        user_id,
        strptime(checkin, '%Y-%m-%d')   as checkin,
        strptime(checkout, '%Y-%m-%d')  as checkout,
        city_id,
        device_class,
        affiliate_id,
        booker_country,
        hotel_country,
        utrip_id
    from raw
    where user_id is not null
      and checkin is not null
      and checkout is not null
      and utrip_id is not null
      and city_id is not null
),

base as (
    select
        row_number() over (
            order by r.utrip_id, r.checkin, r.checkout, r.city_id)  as obs_id,
        md5(concat_ws('|', r.user_id, r.utrip_id, r.checkin, r.checkout,
                      r.city_id, r.device_class, r.affiliate_id,
                      r.booker_country, r.hotel_country))             as obs_key,
        r.user_id                                               as userid,
        r.utrip_id                                               as utrip,
        r.checkin,
        r.checkout,
        date_diff('day', r.checkin, r.checkout)                   as nights,
        date_diff('day', DATE '1970-01-01', r.checkin)             as checkin_day,
        date_diff('day', DATE '1970-01-01', r.checkout)           as checkout_day,
        r.city_id,
        r.device_class,
        r.affiliate_id,
        r.booker_country,
        r.hotel_country,
        coalesce(ct.tier, 'ct_other')                             as hotel_ct,
        coalesce(at_.aff_tier, 'aff_other')                       as aff_tier,
        r.checkout <= r.checkin                                   as qf_bad_dates
    from rows_ok r
    left join {{ ref('country_tiers') }}   ct on ct.country = r.hotel_country
    left join {{ ref('affiliate_tiers') }} at_ on at_.affiliate_id = r.affiliate_id
)

select
    *,
    -- leg index within the trip (1-based by checkin; checkout/city tiebreak
    -- keeps the order deterministic)
    row_number() over (
        partition by utrip order by checkin, checkout, city_id)  as leg_idx,
    'm' || strftime(checkin, '%m')                               as month_cell,
    -- day-of-week, Sunday = 0 (measured near-uniform in the source)
    'd' || strftime(checkin, '%w')                               as dow_cell,
    -- public stay-night class grid; undefined on bad-date rows
    case
        when qf_bad_dates then null
        when nights = 1 then 'n_1'
        when nights = 2 then 'n_2'
        when nights = 3 then 'n_3'
        when nights < {{ var('nights_e7') }} then 'n_4_6'
        else 'n_7p'
    end                                                          as nights_cell,
    -- OVERLAP flag: this leg starts before the previous leg of the same trip
    -- has ended -> the same user would be in two hotels at once. Measured 0
    -- rows in the source data (99.11% of legs are perfectly contiguous);
    -- the flag guards ingestion anyway, and the synthetic round-trip must
    -- pass it with 0 flagged rows.
    coalesce(
        checkin_day < lag(checkout_day) over (
            partition by utrip order by checkin, checkout, city_id),
        false)                                                   as qf_overlap
from base
