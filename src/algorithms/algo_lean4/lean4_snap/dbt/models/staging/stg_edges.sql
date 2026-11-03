-- Canonical undirected friendship edges (src < dst enforced at ingest,
-- re-tested in q_data_integrity).
select
    src::bigint as src,
    dst::bigint as dst
from read_csv(
    '{{ var("ingest_dir") }}/edges.tsv',
    delim = '\t',
    header = true,
    columns = {'src': 'bigint', 'dst': 'bigint'}
)
