-- Episode (journey) rows with public length-bin occupancy: feat_episodes
-- plus how many episodes share each ep_bin — the enclave view the DP
-- episode-length histogram (mart_dp_episode) is clipped from.
select
    e.*,
    count(*) over (partition by ep_bin) as n_in_bin
from {{ ref('feat_episodes') }} e
order by n_hops desc
