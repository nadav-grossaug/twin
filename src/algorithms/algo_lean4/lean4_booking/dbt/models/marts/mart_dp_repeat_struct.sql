-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one USER in SampCert's add/remove-one Neighbour relation.
-- Cell grid: the 6 PUBLIC repeat-structure cells over a user's FIRST repeat
-- trip (a trip with has_repeat_city), classified by its transition anatomy
-- (spec 1.2.0):
--   r_ext1  exactly 1 consecutive same-city pair, no non-adjacent revisit
--   r_ext2  exactly 2 consecutive same-city pairs, no revisit
--   r_ext3p 3+ consecutive same-city pairs, no revisit
--   r_ret1  exactly 1 non-adjacent revisit, no consecutive pair
--   r_ret2p 2+ non-adjacent revisits, no consecutive pair
--   r_both  at least one of each
-- Measured shares (enclave, 288,348 trips): 11.22 / 3.62 / 3.46 / 13.20 /
-- 1.16 / 5.42 %. Users with no repeat trip contribute to NO cell (their
-- mass is the return_rate release's complement), so sensitivity = clip =
-- dp_count_clip = 1. The generator uses this as the CONDITIONAL structure
-- mix given a repeat trip, replacing the single synthetic return hop that
-- under-produced consecutive same-city stays 5.4x (real 9.01% of legs,
-- synth 1.66%).
with cells as (
    select * from (values
        ('r_ext1', 1), ('r_ext2', 2), ('r_ext3p', 3),
        ('r_ret1', 4), ('r_ret2p', 5), ('r_both', 6)
    ) as t(cell, ord)
),

anatomy as (
    select
        f.userid,
        f.utrip,
        count(*) filter (
            where f.prev_city_id = f.next_city_id) as consec,
        count(*) filter (
            where f.prev_city_id != f.next_city_id
              and f.city_return) as revisit
    from {{ ref('feat_transitions') }} f
    group by 1, 2
),

first_repeat as (
    select
        r.userid,
        r.consec,
        r.revisit
    from (
        select
            t.userid,
            a.utrip,
            a.consec,
            a.revisit,
            row_number() over (
                partition by t.userid
                order by t.start_day, t.utrip) as rn
        from {{ ref('feat_trips') }} t
        join anatomy a using (userid, utrip)
        where t.has_repeat_city
    ) r
    where r.rn = 1
),

users as (
    select
        userid,
        case
            when consec = 1  and revisit = 0  then 'r_ext1'
            when consec = 2  and revisit = 0  then 'r_ext2'
            when consec >= 3 and revisit = 0  then 'r_ext3p'
            when consec = 0  and revisit = 1  then 'r_ret1'
            when consec = 0  and revisit >= 2 then 'r_ret2p'
            else 'r_both'
        end as cell
    from first_repeat
)

select
    c.cell,
    coalesce(count(u.userid), 0)::bigint as clipped_count,
    count(u.userid)                      as n_contributing_users,
    {{ var('dp_count_clip') }}           as clip
from cells c
left join users u on u.cell = c.cell
group by c.cell, c.ord
order by c.ord
