-- Top-k TLS SNI server names: COMMON-KNOWLEDGE metadata (popular public
-- hostnames in ClientHellos), released as a suppression-thresholded artifact
-- (release_policy.yaml artifacts.tls_sni_topk), NOT through the epsilon
-- ledger — same status as mart_dns_topk. Suppression: an SNI must appear in
-- TLS flows of >= 2 distinct hosts and >= 10 flows; single-label names
-- (local hostnames) are excluded.
select
    sni_first           as sni,
    count(*)            as n_flows,
    count(distinct sa)  as n_src_hosts
from {{ ref('feat_tls') }}
where sni_first is not null
  and len(string_split(sni_first, '.')) >= 2
group by sni_first
having count(distinct sa) >= 2 and count(*) >= 10
order by n_flows desc, sni
limit 32
