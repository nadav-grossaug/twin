-- Local-clustering class occupancy. cc = cc_num/cc_den per node; classes by
-- exact integer comparison (see feat_nodes).
select
    cc_class,
    count(*)::bigint                                   as n_nodes,
    round(avg(case when cc_den > 0
                   then cc_num / cc_den::double
                   else 0 end), 4)                     as mean_cc
from {{ ref('feat_nodes') }}
group by cc_class
order by min(case cc_class
    when 'cc_zero'  then 0 when 'cc_0_20' then 1 when 'cc_20_40' then 2
    when 'cc_40_60' then 3 when 'cc_60_80' then 4 when 'cc_80_99' then 5
    else 6 end)
