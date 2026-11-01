-- Per-day return features: LOG returns (random-walk modelling basis), the
-- public 5-class return taxonomy, calendar/seasonality fields, rolling
-- volatility + vol state (the GARCH discretization grid), drawdown from the
-- running peak, and interday technical-analysis features (SMA, RSI, momentum).
-- All thresholds are public spec vars.
with p as (
    select
        obs_id,
        ticker,
        d,
        price,
        qf_nonpositive_price,
        lag(price) over w                         as prev_price,
        lag(d) over w                             as prev_d,
        row_number() over w - 1                   as day_idx
    from {{ ref('stg_prices') }}
    window w as (partition by ticker order by d)
),

r as (
    select
        *,
        d - prev_d                                as gap_days,
        case
            when prev_price is not null and prev_price > 0 and price > 0
            then ln(price / prev_price)
        end                                       as log_ret
    from p
),

c as (
    select
        *,
        case when log_ret is not null
             then cast(round(log_ret * 10000) as integer) end as ret_bp
    from r
),

k as (
    select
        *,
        case
            when ret_bp is null then null
            when ret_bp <= -{{ var('ret_big_bp') }}  then 'crash'
            when ret_bp <  -{{ var('ret_flat_bp') }} then 'down'
            when ret_bp <=  {{ var('ret_flat_bp') }} then 'flat'
            when ret_bp <   {{ var('ret_big_bp') }}  then 'up'
            else 'surge'
        end as ret_class,
        case
            when ret_bp is null then null
            when ret_bp >  {{ var('ret_flat_bp') }} then 1
            when ret_bp < -{{ var('ret_flat_bp') }} then -1
            else 0
        end as ret_sign
    from c
),

roll as (
    select
        *,
        stddev_samp(ret_bp) over (
            partition by ticker order by d
            rows between {{ var('volburst_window') - 1 }} preceding and current row
        )                                                     as vol5_bp,
        count(ret_bp) over (
            partition by ticker order by d
            rows between {{ var('volburst_window') - 1 }} preceding and current row
        )                                                     as vol5_n,
        avg(price) over (
            partition by ticker order by d
            rows between 4 preceding and current row
        )                                                     as sma5,
        avg(price) over (
            partition by ticker order by d
            rows between 19 preceding and current row
        )                                                     as sma20,
        sum(ret_bp) over (
            partition by ticker order by d
            rows between 4 preceding and current row
        )                                                     as momentum5_bp,
        avg(greatest(ret_bp, 0)) over (
            partition by ticker order by d
            rows between 13 preceding and current row
        )                                                     as gain14,
        avg(greatest(-ret_bp, 0)) over (
            partition by ticker order by d
            rows between 13 preceding and current row
        )                                                     as loss14,
        max(price) over (
            partition by ticker order by d
            rows between unbounded preceding and current row
        )                                                     as peak
    from k
)

select
    obs_id,
    ticker,
    d,
    day_idx,
    d - date '1970-01-01'                                     as day_num,
    price,
    prev_price,
    gap_days,
    coalesce(gap_days > {{ var('gap_max_days') }}, false)     as qf_gap_break,
    qf_nonpositive_price,
    log_ret,
    ret_bp,
    ret_class,
    ret_sign,
    abs(ret_bp) >= {{ var('shock_bp') }}                      as is_shock,
    -- seasonality fields (public calendar facts)
    isodow(d)                                                 as dow,          -- 1=Mon..5=Fri
    ['mon','tue','wed','thu','fri'][isodow(d)]                as dow_name,
    month(d)                                                  as month,
    strftime(d, '%G-%V')                                      as iso_week,
    -- rolling volatility and the public vol-state grid
    vol5_bp,
    vol5_n,
    case
        when vol5_bp is null or vol5_n < {{ var('volburst_window') }} then null
        when vol5_bp <  {{ var('vol_calm_max_bp') }}     then 'calm'
        when vol5_bp <  {{ var('vol_normal_max_bp') }}   then 'normal'
        when vol5_bp <  {{ var('vol_elevated_max_bp') }} then 'elevated'
        else 'burst'
    end                                                       as vol_state,
    vol5_n >= {{ var('volburst_window') }}
        and vol5_bp >= {{ var('volburst_min_bp') }}           as is_volburst,
    -- technical-analysis features (interday)
    sma5,
    sma20,
    case when sma20 > 0 then price / sma20 - 1 end            as sma20_dev,
    case when sma5 > 0 and sma20 > 0 then sma5 / sma20 - 1 end as sma_cross,
    momentum5_bp,
    case when gain14 + loss14 > 0
         then 100.0 * gain14 / (gain14 + loss14) end          as rsi14,
    -- drawdown from the running peak
    peak,
    case when peak > 0 then 1 - price / peak end              as dd_frac,
    peak > 0 and (1 - price / peak)
        >= {{ var('drawdown_num') }} / {{ var('drawdown_den') }}.0 as in_drawdown
from roll
