-- Canonical daily price observations from the stock TSV.
-- One row per (ticker, trading day); stable obs_id and content-hash obs_key.
with raw as (
    select *
    from read_csv(
        '{{ var("price_path") }}',
        delim = '\t',
        header = true,
        columns = {
            'stockname': 'varchar',
            'date-yyyy-mm-dd': 'date',
            'stockprice': 'double'
        }
    )
),

rows_ok as (
    select
        stockname                as ticker,
        "date-yyyy-mm-dd"        as d,
        stockprice               as price
    from raw
    where stockname is not null
      and "date-yyyy-mm-dd" is not null
      and stockprice is not null
)

select
    row_number() over (order by ticker, d)            as obs_id,
    md5(concat_ws('|', ticker, d, price))             as obs_key,
    ticker,
    d,
    price,
    price <= 0                                        as qf_nonpositive_price,
    isodow(d) > 5                                     as qf_weekend
from rows_ok
