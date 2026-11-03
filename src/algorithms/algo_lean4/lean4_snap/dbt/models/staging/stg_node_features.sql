-- Sparse per-node feature observations (value is always 1: absent means
-- 0/unobserved — the ingest encoding convention). raw_category comes from
-- the SNAP featname prefix; the seed maps it onto the 11 PUBLIC categories
-- (the DP grid half that is fixed independent of the data).
select
    f.node_id::bigint                  as node_id,
    f.feature_key,
    f.raw_category,
    coalesce(m.category, 'other')      as category,
    f.value::bigint                    as value
from read_csv(
    '{{ var("ingest_dir") }}/node_features.tsv',
    delim = '\t',
    header = true,
    columns = {
        'node_id': 'bigint',
        'feature_key': 'varchar',
        'raw_category': 'varchar',
        'value': 'bigint'
    }
) f
left join {{ ref('category_map') }} m on m.raw_category = f.raw_category
