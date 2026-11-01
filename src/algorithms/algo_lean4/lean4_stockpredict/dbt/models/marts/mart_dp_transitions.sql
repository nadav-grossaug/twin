-- DP input: return-class transition counts on the public 5x5 grid
-- ('cur>nxt' cells), uniform per-ticker per-cell clip dp_clip. The distilled
-- semi-Markov replacement for the NextClassGRU teacher; gap-broken upstream.
with cells as (
    select a.c || '>' || b.c as cell
    from (select unnest(['crash','down','flat','up','surge']) as c) a
    cross join (select unnest(['crash','down','flat','up','surge']) as c) b
),

seq as (
    select
        ticker,
        ret_class                 as cur,
        lead(ret_class) over w    as nxt,
        lead(d) over w - d        as gap_days
    from {{ ref('feat_returns') }}
    where ret_class is not null
    window w as (partition by ticker order by d)
),

counts as (
    select ticker, cur || '>' || nxt as cell, count(*) as n
    from seq
    where nxt is not null and gap_days <= {{ var('gap_max_days') }}
    group by 1, 2
)

select
    c.cell,
    coalesce(sum(least(n, {{ var('dp_clip') }})), 0)::bigint as clipped_count,
    count(distinct case when n > 0 then ticker end)          as n_contributing_users,
    {{ var('dp_clip') }}                                     as clip
from cells c
left join counts using (cell)
group by c.cell
order by c.cell
