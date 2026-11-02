-- Turning-angle histogram over the 8 public 45-degree bins, moving points
-- only. tb_k covers [-180 + 45k, -180 + 45(k+1)). The enclave ledger behind
-- the category-2 turn release; the generator's correlated-heading walk
-- draws its turn innovations from the released weights.
with bins as (
    select 'tb_' || i::varchar as turn_bin, i as ord
    from range(8) t(i)
)

select
    b.turn_bin,
    coalesce(count(s.obs_id), 0)::bigint   as n_points,
    count(distinct s.user_id)              as n_users
from bins b
left join (
    select obs_id, user_id, turn_bin
    from {{ ref('feat_steps') }}
    where is_moving and turn_bin is not null and mode_class != 'XX'
) s using (turn_bin)
group by b.turn_bin, b.ord
order by b.ord
