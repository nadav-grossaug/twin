-- Within-trajectory mode-leg chains (walk -> bus -> walk ...): first-order
-- transitions between consecutive legs of one trajectory, chained only when
-- the inter-leg gap is <= sp_stream_break_s (recording-continuous).
with legs as (
    select
        user_id,
        trajectory_id,
        mode_class,
        ts_start_s,
        ts_end_s,
        lag(mode_class) over w as prev_mode,
        lag(ts_end_s)   over w as prev_end_s
    from {{ ref('feat_legs') }}
    where mode_class != 'XX'
    window w as (partition by user_id, trajectory_id order by ts_start_s)
),

trans as (
    select prev_mode, mode_class as next_mode, count(*) as n
    from legs
    where prev_mode is not null
      and ts_start_s - prev_end_s <= {{ var('sp_stream_break_s') }}
    group by prev_mode, mode_class
),

cur_tot as (
    select prev_mode, sum(n) as n_cur from trans group by prev_mode
),

nxt_marg as (
    select next_mode, sum(n)::double / (select sum(n) from trans) as p_marg
    from trans group by next_mode
)

select
    t.prev_mode,
    t.next_mode,
    t.n,
    t.n::double / c.n_cur              as p_next,
    (t.n::double / c.n_cur) / m.p_marg as lift
from trans t
join cur_tot c using (prev_mode)
join nxt_marg m using (next_mode)
order by t.n desc
