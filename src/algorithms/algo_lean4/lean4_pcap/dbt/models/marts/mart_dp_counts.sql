-- Input to the differentially private release (NOT itself safe to export).
-- Privacy unit: one MONITORED (internal) host = one "user" in SampCert's
-- add/remove-one Neighbour relation; each flow is attributed to exactly one
-- internal endpoint (sa when src_internal, else da), so external responders
-- are counted through the monitored host they talked to, never as users.
-- Cell grid: the PUBLIC 7-class generator service taxonomy (genServices).
-- Contribution bounding is PROPORTIONAL (L1): each host's cell vector is its
-- own traffic mix allocated over a total budget of dp_clip_total flows
-- (largest-remainder rounding; hosts with fewer flows contribute their actual
-- counts). The vector sums to <= dp_clip_total and no cell exceeds it, so
-- changing one user moves each cell by at most dp_clip_total — the per-cell
-- sensitivity the Lean mechanism (lake exe release) is calibrated to.
-- Rationale (2026-10-31): uniform per-cell clipping at 4 saturated every
-- major class for the 15 internal hosts while 2,433 external one-flow
-- sources dominated ephemeral/other — the released mix was inverted vs the
-- capture. Proportional allocation preserves each host's mix shape.
with cells as (
    select unnest([
        'http', 'https', 'dns', 'ntp', 'chat', 'ephemeral', 'other'
    ]) as cell
),

flows as (
    select
        case when src_internal then sa else da end as host,
        case when service_class in ('http','https','dns','ntp','chat','ephemeral')
             then service_class else 'other' end as cell
    from {{ ref('feat_flows') }}
    where src_internal or dst_internal
),

counts as (
    select host, cell, count(*) as n
    from flows
    group by host, cell
),

totals as (
    select
        host,
        sum(n) as total,
        least(sum(n), {{ var('dp_clip_total') }}) as budget
    from counts
    group by host
),

alloc as (
    select
        c.host,
        c.cell,
        (c.n * t.budget) // t.total as base,
        (c.n * t.budget) % t.total  as frac,
        t.budget
    from counts c
    join totals t using (host)
),

ranked as (
    select
        *,
        row_number() over (partition by host order by frac desc, cell) as rk,
        budget - sum(base) over (partition by host) as remainder
    from alloc
),

per_user as (
    select
        host,
        cell,
        (base + case when rk <= remainder then 1 else 0 end)::bigint as clipped_n
    from ranked
)

select
    c.cell,
    coalesce(sum(p.clipped_n), 0)::bigint  as clipped_count,
    count(distinct case when p.clipped_n > 0 then p.host end) as n_contributing_users,
    {{ var('dp_clip_total') }}             as clip
from cells c
left join per_user p on p.cell = c.cell
group by c.cell
order by c.cell
