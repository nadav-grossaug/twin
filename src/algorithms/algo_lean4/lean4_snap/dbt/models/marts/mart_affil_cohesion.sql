-- Affiliation cohesion: category x group-size bin -> how many key-groups,
-- their mean internal density and mean size. The random-pair baseline for
-- density is the global edge density 88,234 / C(4039,2) ~= 0.0108.
-- ENCLAVE-ONLY (derived from key-level groups; policy denied list). The
-- `suppressed` flag marks cells below the spec suppression threshold for
-- display discipline — nothing here is released either way.
select
    category,
    case
        when n_nodes = 2 then 'g_2'
        when n_nodes <= 4 then 'g_3_4'
        when n_nodes <= 8 then 'g_5_8'
        when n_nodes <= 16 then 'g_9_16'
        else 'g_17p'
    end                                   as size_bin,
    count(*)::bigint                      as n_groups,
    avg(density)                          as mean_density,
    avg(n_nodes)                          as mean_size,
    (count(*) < {{ var('suppress_min_nodes') }}) as suppressed
from {{ ref('mart_key_groups') }}
group by 1, 2
order by 1, 2
