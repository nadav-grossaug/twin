-- Edge mixing between rank-labeled communities (c1..c8, rest): the raw
-- (non-DP) counterpart of mart_dp_mixing, plus the within-community share.
select
    pair_cell,
    count(*)::bigint                                        as n_edges,
    sum(case when same_community then 1 else 0 end)::bigint as n_within,
    round(count(*) / (select count(*)::double
                      from {{ ref('feat_edges') }}), 6)     as share
from {{ ref('feat_edges') }}
group by pair_cell
order by n_edges desc
