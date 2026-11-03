-- Community sizes (Louvain, seeded at ingest) with rank labels and public
-- size bins. Rank->member mapping stays enclave-side; only the size
-- histogram and rank-labeled mixing are DP-released.
select
    f.community_id,
    f.community_rank,
    count(*)::bigint as n_nodes,
    case
        when count(*) < {{ var('csize_e25') }}  then 'csize_lt25'
        when count(*) < {{ var('csize_e51') }}  then 'csize_25_50'
        when count(*) < {{ var('csize_e101') }} then 'csize_51_100'
        when count(*) < {{ var('csize_e201') }} then 'csize_101_200'
        when count(*) < {{ var('csize_e401') }} then 'csize_201_400'
        else 'csize_gt400'
    end as size_bin
from {{ ref('feat_nodes') }} f
group by f.community_id, f.community_rank
order by n_nodes desc
