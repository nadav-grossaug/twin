{% macro dp_hist(cells, events_sql, clip_var) %}
-- Shared DP-input shape (NOT itself safe to export). Privacy unit: one USER
-- (add/remove-one Neighbour; their whole trace is one record). Contribution
-- bounding is PROPORTIONAL (L1) over least(n_events, {{ clip_var }}) with
-- largest-remainder rounding, so per-cell sensitivity equals the per-user
-- total and the proved mechanism applies unchanged (METHOD §6: uniform
-- per-cell clipping flattens the mix — not used).
with cells as (
    select * from (values
        {%- for c in cells %}
        ('{{ c }}', {{ loop.index }}){{ "," if not loop.last }}
        {%- endfor %}
    ) as t(cell, ord)
),

events as (
    {{ events_sql }}
),

counts as (
    select user_id, cell, count(*) as n
    from events
    group by user_id, cell
),

totals as (
    select
        user_id,
        sum(n) as total,
        least(sum(n), {{ var(clip_var) }}) as budget
    from counts
    group by user_id
),

alloc as (
    select
        c.user_id,
        c.cell,
        (c.n * t.budget) // t.total as base,
        (c.n * t.budget) % t.total  as frac,
        t.budget
    from counts c
    join totals t using (user_id)
),

ranked as (
    select
        *,
        row_number() over (partition by user_id order by frac desc, cell) as rk,
        budget - sum(base) over (partition by user_id) as remainder
    from alloc
),

per_user as (
    select
        user_id,
        cell,
        (base + case when rk <= remainder then 1 else 0 end)::bigint as clipped_n
    from ranked
)

select
    c.cell,
    coalesce(sum(p.clipped_n), 0)::bigint                        as clipped_count,
    count(distinct case when p.clipped_n > 0 then p.user_id end) as n_contributing_users,
    {{ var(clip_var) }}                                          as clip
from cells c
left join per_user p on p.cell = c.cell
group by c.cell, c.ord
order by c.ord
{% endmacro %}
