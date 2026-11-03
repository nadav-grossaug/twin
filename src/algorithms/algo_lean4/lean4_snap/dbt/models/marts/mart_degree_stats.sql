-- Degree distribution over the public bins + per-class summary stats.
select
    degree_class,
    count(*)::bigint        as n_nodes,
    min(degree)             as min_degree,
    median(degree)          as median_degree,
    round(avg(degree), 2)   as mean_degree,
    max(degree)             as max_degree
from {{ ref('feat_nodes') }}
group by degree_class
order by min(degree)
