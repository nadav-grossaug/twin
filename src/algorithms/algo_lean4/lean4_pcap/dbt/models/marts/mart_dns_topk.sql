-- Top-k most-queried DNS names: COMMON-KNOWLEDGE metadata (popular public
-- domains behave like the fixed service taxonomy), released as a
-- suppression-thresholded artifact (release_policy.yaml artifacts.dns_topk),
-- NOT through the epsilon ledger. Suppression: a name must be queried by >= 2
-- distinct hosts and >= 10 times, so single-host private names never qualify
-- (on this capture names spread over at most 2-3 hosts: 8,817 names at 2).
-- Reverse-lookup names (*.arpa) are excluded outright: they encode internal
-- IP octets by construction, whatever their popularity. Single-label names
-- (local hostnames, no public TLD) are site-private and excluded too.
select
    qname,
    count(*)            as n_queries,
    count(distinct sa)  as n_src_hosts
from {{ ref('feat_dns') }}
where not is_reverse_lookup and n_labels >= 2
group by qname
having count(distinct sa) >= 2 and count(*) >= 10
order by n_queries desc, qname
limit 32
