-- Triadic closure conditional: P(closed | wedge-center degree class).
-- Wedges centered at v: wedge_count = d(d-1)/2; closed wedges centered at
-- v = tri_count (each triangle through v closes exactly the one wedge
-- centered at v). This grid is the sequence-analog of the build (the
-- "flow" here is wedge -> triangle, not time) and the generator's
-- clustering driver.
select
    degree_class,
    count(*)::bigint                            as n_nodes,
    sum(wedge_count)::bigint                    as wedges,
    sum(tri_count)::bigint                      as closed,
    sum(wedge_count - tri_count)::bigint        as open,
    round(sum(tri_count)
          / nullif(sum(wedge_count), 0)::double, 4) as closure_rate
from {{ ref('feat_nodes') }}
group by degree_class
order by min(degree)
