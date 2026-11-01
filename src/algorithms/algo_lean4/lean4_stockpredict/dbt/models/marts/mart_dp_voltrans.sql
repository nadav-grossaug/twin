-- DP input: GARCH discretization — vol-state transition counts on the public
-- 4x4 grid ('cur>nxt' cells over calm/normal/elevated/burst), uniform
-- per-ticker per-cell clip dp_clip. Volatility clustering (the GARCH
-- footprint) survives as diagonal mass in this histogram.
with cells as (
    select a.s || '>' || b.s as cell
    from (select unnest(['calm','normal','elevated','burst']) as s) a
    cross join (select unnest(['calm','normal','elevated','burst']) as s) b
),

seq as (
    select
        ticker,
        vol_state                 as cur,
        lead(vol_state) over w    as nxt,
        lead(d) over w - d        as gap_days
    from {{ ref('feat_returns') }}
    where vol_state is not null
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
