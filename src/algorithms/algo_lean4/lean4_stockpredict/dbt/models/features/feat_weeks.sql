-- Weekly shape structure: the Mon-Fri return-class vector per ticker per ISO
-- week (the nested-structure analog of the reference repo's TLS record
-- sequences), plus weekly aggregates for seasonality analysis.
select
    ticker,
    iso_week,
    count(ret_bp)                                   as n_days,
    count(ret_bp) = 5                               as complete_week,
    sum(ret_bp)                                     as week_ret_bp,
    stddev_samp(ret_bp)                             as week_vol_bp,
    -- class initials in day order, e.g. 'udfus' (c=crash d=down f=flat u=up s=surge)
    string_agg(substr(ret_class, 1, 1), '' order by d) as shape,
    min(d)                                          as week_start,
    list(ret_class order by d)                      as classes
from {{ ref('feat_returns') }}
where ret_bp is not null
group by ticker, iso_week
order by ticker, iso_week
