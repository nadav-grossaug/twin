-- Static-snapshot assertion: the brief asked whether input records follow a
-- "timely flow" logic. The measured answer is NO — the dataset carries no
-- usable temporal fields. Attribute names that LOOK temporal (birthday,
-- work;start_date, work;end_date, education;year) are anonymized BINARY
-- presence flags: every feature value is exactly 1 (sparse encoding), so
-- nothing orders the records in time. This model proves that claim as
-- counters: nonbinary_feature_values and timelike_nonbinary_values must be
-- 0, making any sequence/temporal layer inapplicable by measurement, not by
-- assumption (see wiki/dataset-snap.md). The sequence analog in this build
-- is the triadic-closure process (mart_wedge_closure).
select 'feature_values_total' as counter, count(*)::bigint as n
from {{ ref('stg_node_features') }}
union all
select 'nonbinary_feature_values', sum(case when value != 1 then 1 else 0 end)::bigint
from {{ ref('stg_node_features') }}
union all
select 'timelike_feature_keys', count(distinct feature_key)::bigint
from {{ ref('stg_node_features') }}
where regexp_matches(feature_key, '(birthday|year|date|month|day|time)')
union all
select 'timelike_nonbinary_values', sum(case when value != 1 then 1 else 0 end)::bigint
from {{ ref('stg_node_features') }}
where regexp_matches(feature_key, '(birthday|year|date|month|day|time)')
