-- Releasable affiliation-cohesion rollup (suppression-thresholded ONLY — no
-- formal guarantee; the policy artifact affil_cohesion_stats exports it with
-- min_nodes applied to n_member_nodes). Restricted to the released affiliation
-- categories (spec.py AFFIL_RELEASE_CATS; sync-tested) over the PUBLIC affil size bins; the
-- full per-category table (mart_affil_cohesion) and the key-level groups
-- (mart_key_groups) stay on the denied list.
-- density_pct is an INTEGER percent so every released weight is an exact
-- rational (den 100) for the Lean generator.
{% set affil_cats = ['birthday', 'education', 'gender', 'hometown', 'languages', 'last_name', 'locale', 'location', 'work'] %}
select
    category,
    case
        when n_nodes < {{ var('affil_g5') }} then 'g_2_4'
        when n_nodes < {{ var('affil_g9') }} then 'g_5_8'
        when n_nodes < {{ var('affil_g17') }} then 'g_9_16'
        else 'g_17p'
    end                                       as size_bin,
    count(*)::bigint                          as n_groups,
    sum(n_nodes)::bigint                      as n_member_nodes,
    round(100 * avg(density))::bigint         as density_pct
from {{ ref('mart_key_groups') }}
where category in ({% for c in affil_cats %}'{{ c }}'{{ ", " if not loop.last else "" }}{% endfor %})
group by 1, 2
order by 1, 2
