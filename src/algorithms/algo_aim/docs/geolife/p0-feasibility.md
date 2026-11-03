# P0 feasibility: public-map layer (L0)

Run on this DevVision machine, 2026-11-01. Python 3.11 venv `C:/Users/NaomiD/geo-venv`, packages via JFrog (`uv pip`).

## 1. Overture reachability: WORKS (no firewall block)

- `https://overturemaps-us-west-2.s3.amazonaws.com/?list-type=2&prefix=release/&delimiter=/` returns the release list
  (2026-09-23.0, 2026-09-23.1, 2026-10-21.0). Anonymous S3 over HTTPS works; no Azure test needed.
- DuckDB 2.0 `httpfs` installs and loads (about 19 s first time). The `spatial` extension FAILED to download
  (`Failed to download extension "spatial" at URL "https://extensions.duckdb.org/v2.0.0/windows_amd64/spatial.duckdb_extension.gz"`),
  i.e. extensions.duckdb.org is blocked. Not needed: Overture parquet has a `bbox` struct (pushdown) and
  WKB `geometry`, which we decode with shapely.
- Release 2026-10-21.0, bbox 116.28..116.32 x 39.96..40.00 (Beijing, Haidian), query by `bbox.*` columns:

| type | rows | wall time |
|---|---|---|
| transportation/connector | 3,992 | 9-11 s |
| transportation/segment | 2,858 (2,788 road, 70 rail) | 155-168 s (wide columns) |
| places/place | 433 | 25 s |
| buildings/building | 5,603 | 91 s |

  Segment classes: footway 739, tertiary 490, residential 476, service 458, secondary 155, steps 116, primary 91, trunk 90 ...
  All segment `sources` are OpenStreetMap. Places: meta 277, Foursquare 127, AllThePlaces 20, Microsoft 7.
  Note: the `categories` column is not present under that name in this release (taxonomy change); inspect schema before
  relying on the category taxonomy (task for the POI step). Only `names`, `bbox` and `sources` were used here.
- Cost to be aware of: a whole city is minutes per type; select only needed columns for segments to speed up.
- Air-gap fallback (not tested; the air-gapped side is out of scope here): download the same
  `theme=*/type=*/` parquet files for the city bbox on DevVision (via `load_overture`, cached to parquet), checksum,
  transfer through the promotion path, and on the other side call `load_overture(..., root="<dir>")` where `<dir>`
  contains `theme=transportation/type=segment/*.parquet` etc., or read the cached parquet directly. `duckdb` and
  local reads need no network and no spatial extension.

## 2. Routable graph: WORKS

`src/geolife/outside/map/overture.py` (`load_overture(ftype, bbox, cache_dir, release, root)` -> pandas DataFrame, parquet cache keyed by
release/type/bbox) and `src/geolife/transfer/map/graph.py` (`build_graph`, `nearest_node`, `shortest_path`).
Graph: nodes = connector ids; each segment is split at its `connectors[].at` fractions into edges with haversine length.
Undirected (one-way / `access_restrictions` / `prohibited_transitions` ignored; add later if routing realism matters).

On the real bbox (road subtype only): 3,992 nodes, 5,174 edges, built in 2.5 s; 258 connected components, largest 3,483
nodes (87%) - restrict to the largest component (or drivable classes) for routing. Shortest path between
(116.29,39.97) and (116.31,39.99): 43 segments, 3,670 m, 20 ms. Nearest-node snapping is O(N) brute force (fine for the
spike; use a KD-tree/STRtree for bulk).

## 3. Map-matching: recommendation

| option | verdict |
|---|---|
| Pure-python HMM (Newson-Krumm) over this graph | Recommended for the spike and the MVP. No install risk, works air-gapped, same graph as generation, fully testable. |
| FMM (C++) | Fast and accurate, but needs a native build (cmake, GDAL, boost) and OSM-format network; not obtainable via JFrog pip/npm path. Revisit only if volume demands it. |
| Valhalla / OSRM | Need an OSM PBF extract plus a prebuilt routing server (Docker image from the `base-images` repo, not Docker Hub), and an equivalent on the air-gapped side. Heaviest; they match against OSM, not the Overture ids we want to emit. Reject unless tokenization quality proves poor. |

`src/geolife/transfer/map/match.py` `match(G, points, radius, sigma, beta, k)`: STRtree candidate search (edges within radius, k nearest),
Gaussian emission, exponential transition on |route distance - great-circle distance| using cutoff Dijkstra, Viterbi.
Tested on a synthetic noisy grid path (exact recovery). On the real graph (120 points every ~30 m along a shortest path, ~5 m
Gaussian noise, default parameters): 1.5 s, matched 52 segments vs 27 true; 19 of 27 true segments recovered, 33 extra.
Extras are mostly short intersection pieces and parallel footway/cycleway/carriageway segments (the graph includes sidewalks).
Expect better results by: restricting to vehicle-usable classes (or per-mode class filters), merging per-segment pieces, a
turn penalty, and tuning sigma/beta on GeoLife (noisier than 5 m). Cost is O(T*K^2) Dijkstras; no benchmark for millions of
points - batch/cache single-source results per node (already cached within one call).

## 4. Datum (WGS-84 vs GCJ-02)

- GeoLife is WGS-84 (Microsoft). Overture roads here are 100% OpenStreetMap-sourced; OSM policy is WGS-84 (known risk: old
  imports of GCJ-02-shifted data in China, locally). Places come from meta/Foursquare/others, whose Chinese data could in
  principle be GCJ-02.
- Size of the problem: the public WGS-84 -> GCJ-02 formula gives a shift of about 520 m east and 140 m north (~540 m total) at the
  bbox (computed at 116.30,39.98 / 116.32,40.0 / 116.397,39.908: 538 / 542 / 555 m). So it is not a 100 m effect but ~5x
  larger; a ~100 m offset check is easily feasible (resolution is limited by road/POI position accuracy, ~10-20 m).
- Spot check done: Overture place "天安门城楼" is at (116.39137, 39.90659), consistent with WGS-84 (about 39.907, 116.391);
  the GCJ-02 position would be about (116.397, 39.908). "天安门广场" (116.39166, 39.90333) is likewise WGS-84-like. One point
  is weak evidence but supports no-shift.
- Concrete procedure (do it once the GeoLife sample is available on DevVision):
  1. Take GeoLife points of one dense city area (e.g. Haidian, ~100k points) and the Overture road graph for the bbox.
  2. For candidate shifts (dE, dN) in a grid of -700..700 m, step 20 m (and the analytic WGS->GCJ shift), apply to the GPS
     points and compute the median (and 25th percentile) distance from each point to the nearest segment, using the vehicle
     classes only.
  3. The minimum should be at (0,0) with median ~10-20 m if the datums agree; a clear minimum near (+520 E, +140 N) (or
     -520/-140 when shifting the other way) would mean the map is GCJ-02.
  4. Repeat for 20-30 POI landmarks of known WGS-84 coordinates (Wikipedia/OSM) against Overture place positions.
  5. If shifted, convert Overture to WGS-84 with the inverse GCJ-02 iteration before building the graph.

## Files

- `src/geolife/outside/map/overture.py`, `graph.py`, `match.py`
- `tests/geolife/test_map_core.py` (4 tests, no network; run `PYTHONPATH=src python -m pytest tests/geolife/test_map_core.py`)
- Dependencies used: duckdb (lazy import, only for download), pandas, pyarrow, networkx, shapely.
