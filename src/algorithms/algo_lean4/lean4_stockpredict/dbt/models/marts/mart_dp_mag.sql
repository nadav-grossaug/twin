-- DP input: |log return| magnitude histogram over PUBLIC bin edges (bp):
-- mag_0 [0,50]  mag_1 [51,150]  mag_2 [151,299]  mag_3 [300,599]
-- mag_4 [600,1199]  mag_5 [1200,+). Uniform per-ticker per-cell clip dp_clip.
-- The bins nest inside the class taxonomy (mag_0 = flat band; 1-2 = down/up;
-- 3-5 = crash/surge) so the generator can draw magnitudes per class.
with cells as (
    select unnest(['mag_0','mag_1','mag_2','mag_3','mag_4','mag_5']) as cell
),

counts as (
    select
        ticker,
        case
            when abs(ret_bp) <= {{ var('ret_flat_bp') }} then 'mag_0'
            when abs(ret_bp) <= 150                      then 'mag_1'
            when abs(ret_bp) <  {{ var('ret_big_bp') }}  then 'mag_2'
            when abs(ret_bp) <  600                      then 'mag_3'
            when abs(ret_bp) <  1200                     then 'mag_4'
            else 'mag_5'
        end as cell,
        count(*) as n
    from {{ ref('feat_returns') }}
    where ret_bp is not null
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
