-- Shared-feature profile of every FRIENDSHIP edge: how many anonymized
-- feature keys the two endpoints have in common, per public category, plus
-- the total. One row per canonical edge (zero-share edges included).
-- ENCLAVE-ONLY (edge-level, person-identifying; in the policy denied list).
-- Wide format: one column per public category (stable column set beats a
-- long table for the affiliation minimodels that consume this).
with shared as (
    select
        e.src,
        e.dst,
        f1.category,
        count(*) as n_shared
    from {{ ref('stg_edges') }} e
    join {{ ref('stg_node_features') }} f1 on f1.node_id = e.src
    join {{ ref('stg_node_features') }} f2
      on f2.node_id = e.dst
     and f2.feature_key = f1.feature_key
    group by e.src, e.dst, f1.category
)

select
    e.src,
    e.dst,
    coalesce(sum(case when s.category = 'birthday' then s.n_shared end), 0)::bigint   as shared_birthday,
    coalesce(sum(case when s.category = 'education' then s.n_shared end), 0)::bigint  as shared_education,
    coalesce(sum(case when s.category = 'first_name' then s.n_shared end), 0)::bigint as shared_first_name,
    coalesce(sum(case when s.category = 'gender' then s.n_shared end), 0)::bigint     as shared_gender,
    coalesce(sum(case when s.category = 'hometown' then s.n_shared end), 0)::bigint   as shared_hometown,
    coalesce(sum(case when s.category = 'languages' then s.n_shared end), 0)::bigint  as shared_languages,
    coalesce(sum(case when s.category = 'last_name' then s.n_shared end), 0)::bigint  as shared_last_name,
    coalesce(sum(case when s.category = 'locale' then s.n_shared end), 0)::bigint     as shared_locale,
    coalesce(sum(case when s.category = 'location' then s.n_shared end), 0)::bigint   as shared_location,
    coalesce(sum(case when s.category = 'work' then s.n_shared end), 0)::bigint       as shared_work,
    coalesce(sum(case when s.category = 'other' then s.n_shared end), 0)::bigint      as shared_other,
    coalesce(sum(s.n_shared), 0)::bigint                                              as total_shared
from {{ ref('stg_edges') }} e
left join shared s on s.src = e.src and s.dst = e.dst
group by e.src, e.dst
