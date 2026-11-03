-- Circle-membership histogram over the public bins.
select
    circle_class,
    count(*)::bigint as n_nodes,
    round(avg(n_circles), 3) as mean_circles
from {{ ref('feat_nodes') }}
group by circle_class
order by min(n_circles)
