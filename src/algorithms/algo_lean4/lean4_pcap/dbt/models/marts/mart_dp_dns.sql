-- Input to the DP release of the DNS name-shape mix (NOT itself exportable).
-- Privacy unit: one source host (sa), as in mart_dp_counts. Cells are a fixed
-- PUBLIC shape taxonomy over query records — kind (ordinary / reverse lookup /
-- service discovery), label-count band, and name-length band. No qname, label
-- or domain string ever reaches the mechanism. One query contributes to one
-- cell per facet; per-cell sensitivity stays dp_clip and the budget ledger
-- charges every cell.
with cells as (
    select unnest(['kind_ordinary', 'kind_reverse_lookup', 'kind_service_discovery',
                   'labels_2', 'labels_3', 'labels_4', 'labels_5plus',
                   'len_le_24', 'len_gt_24']) as cell
),

q as (
    select
        sa,
        case when is_service_discovery then 'kind_service_discovery'
             when is_reverse_lookup    then 'kind_reverse_lookup'
             else 'kind_ordinary' end                          as kind_cell,
        case when n_labels <= 2 then 'labels_2'
             when n_labels = 3  then 'labels_3'
             when n_labels = 4  then 'labels_4'
             else 'labels_5plus' end                           as label_cell,
        case when qname_len <= 24 then 'len_le_24'
             else 'len_gt_24' end                              as len_cell
    from {{ ref('feat_dns') }}
),

per_user as (
    select sa, cell, least(count(*), {{ var('dp_clip') }}) as clipped_n
    from (
        select sa, kind_cell  as cell from q
        union all
        select sa, label_cell as cell from q
        union all
        select sa, len_cell   as cell from q
    )
    group by sa, cell
)

select
    c.cell,
    coalesce(sum(p.clipped_n), 0)::bigint  as clipped_count,
    count(p.sa)                            as n_contributing_users,
    {{ var('dp_clip') }}                   as clip
from cells c
left join per_user p on p.cell = c.cell
group by c.cell
order by c.cell
