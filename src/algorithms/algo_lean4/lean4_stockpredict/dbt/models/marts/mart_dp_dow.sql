-- DP input: SEASONALITY — (day-of-week, return-class) counts on the public
-- 5x5 grid ('dow|class' cells), uniform per-ticker per-cell clip dp_clip.
with cells as (
    select a.w || '|' || b.c as cell
    from (select unnest(['mon','tue','wed','thu','fri']) as w) a
    cross join (select unnest(['crash','down','flat','up','surge']) as c) b
),

counts as (
    select ticker, dow_name || '|' || ret_class as cell, count(*) as n
    from {{ ref('feat_returns') }}
    where ret_class is not null
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
