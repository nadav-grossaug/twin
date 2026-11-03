-- One row per (node, ego network, circle) membership.
select
    node_id::bigint as node_id,
    ego_id::bigint  as ego_id,
    circle_label
from read_csv(
    '{{ var("ingest_dir") }}/circles.tsv',
    delim = '\t',
    header = true,
    columns = {
        'node_id': 'bigint',
        'ego_id': 'bigint',
        'circle_label': 'varchar'
    }
)
