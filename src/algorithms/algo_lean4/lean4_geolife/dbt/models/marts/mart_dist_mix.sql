-- Trip-displacement mix (plain counts per public class, with user counts for
-- suppression): the enclave ledger behind the category-2 displacement
-- release. Dropped from the DP ledger deliberately — at n = 64 its budget
-- buys only noise (pre-noise SNR simulation); released with suppression
-- instead and declared non-DP.
select
    dist_class,
    count(*)                 as n_trips,
    count(distinct user_id)  as n_users
from {{ ref('feat_visits') }}
where dist_class is not null
group by dist_class
order by n_trips desc
