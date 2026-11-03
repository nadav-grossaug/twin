# Technical contract — hackathon demo

## Stack

| Layer | Choice | Reason |
|---|---|---|
| Shell | Single `index.html` + a few JS/CSS files, no build step | Zero setup, `python -m http.server 8080` and done |
| Graph rendering | [Sigma.js v2](https://www.sigmajs.org/) via CDN | Handles 300-node graphs in the browser without layout lag; reads GraphML or plain JSON |
| Map | [Leaflet.js](https://leafletjs.com/) via CDN | Lightest option for polyline overlays; no API key needed with OSM tiles |
| Charts | [Chart.js](https://www.chartjs.org/) via CDN | Minimal, readable, handles bar + histogram out of the box |
| Data format | JSON for graphs and stats; GeoJSON for traces | Human-readable, no parsing lib needed |

All libs loaded from CDN; no `npm install`, no bundler.

---

## File layout

```
src/presentation/
  plan/           ← these three documents
  app/
    index.html    ← entry point; tab switcher
    graphs.html   ← (or inline section) graph tab
    mobility.html ← (or inline section) mobility tab
    js/
      graphs.js   ← Sigma init, attribute selector, chart renders
      mobility.js ← Leaflet init, layer toggle, city selector, chart renders
    data/
      graph_real_subset.json      ← ~300-node subgraph, nodes have x/y layout + attribute
      graph_synthetic_subset.json ← matching synthetic subgraph
      graph_stats.json            ← degree hist, mixing matrix, discriminator importance (all variants)
      traces_real_beijing.geojson ← 10-15 sampled real user traces (trimmed, not full GPS)
      traces_synth_beijing.geojson
      traces_synth_amsterdam.geojson
      mobility_stats.json         ← outing length hist, revisit rate (real vs synthetic)
```

---

## Data preparation (Python scripts, run once)

Scripts live at `src/presentation/app/data/prepare/` — separate from the pipeline, read-only access to existing outputs.

### `prepare_graph.py`

Input: `snap/inside/data/processed/attributed_graph.npz` (real) and one synthetic graph from `snap/outside/synthetic_v1_1/` (seed 0, eps 4).

Steps:
1. Find the ego neighbourhood of the highest-degree node; take its 1-hop + 2-hop neighbours, trim to 300 nodes by degree.
2. Run `networkx.spring_layout` once, save positions.
3. Export `graph_real_subset.json` and `graph_synthetic_subset.json`: `{ nodes: [{id, x, y, degree, school, hometown, location}], edges: [{source, target}] }`.
4. Compute degree histogram (log bins), attribute mixing (same/cross fractions for school, hometown, location) and discriminator importance (from `snap/eval/results_v1_1/discriminator_importance_v1_1.csv`); write `graph_stats.json`.

### `prepare_mobility.py`

Input: real outings parquet from `geolife/transfer/artifacts/` (or `geolife/inside/data/`); synthetic outings from `geolife/outside/synthetic/`.

Steps:
1. Sample 12 users from the real data; for each user take up to 10 outings; clip to the Beijing bounding box. Write `traces_real_beijing.geojson` (LineString features, property `user_id`).
2. Same for 12 synthetic users from seed 0. Write `traces_synth_beijing.geojson`.
3. If Amsterdam synthetic data exists, write `traces_synth_amsterdam.geojson` similarly; otherwise omit the city selector.
4. Compute outing-length histogram (bins 0–5 km, 5–15, 15–30, 30+) and revisit rate (fraction of outings returning to a previously visited H3-8 cell). Write `mobility_stats.json`.

All scripts run with `python3 -I prepare_graph.py` etc. from the `data/prepare/` directory, with `PYTHONPATH` set to `src`.

---

## Graph JSON schema

```json
{
  "nodes": [
    { "id": "42", "x": 0.3, "y": -0.7, "degree": 15,
      "school": "0", "hometown": "2", "location": "1" }
  ],
  "edges": [
    { "source": "42", "target": "17" }
  ]
}
```

Attribute values are integer strings matching the artifact's category encoding. A `label_map` key maps them to display names if available.

---

## Chart data schema (`graph_stats.json`)

```json
{
  "degree_hist": {
    "bins": [1, 2, 4, 8, 16, 32, 64, 128],
    "real": [120, 95, ...],
    "synthetic": [115, 90, ...],
    "bter": [100, 88, ...]
  },
  "mixing": {
    "school":    { "real_same": 0.59, "synth_same": 0.42 },
    "hometown":  { "real_same": 0.55, "synth_same": 0.38 },
    "location":  { "real_same": 0.52, "synth_same": 0.41 }
  },
  "discriminator": {
    "attributes": 0.43,
    "degree":     0.18,
    "structure":  0.05
  }
}
```

---

## Constraints

- No network calls during the demo — all data is local static files.
- Graph layout pre-computed (positions in JSON) — no physics on page load.
- OSM tile layer requires internet; if the venue has no wifi, swap for a static map image background in Leaflet.
- Real GPS traces: clip to the H3-6 cell covering Beijing central (not sub-city precision) before export — do not export raw GPS coordinates from the real dataset into a browser-accessible file.
- Total `data/` directory should be under 5 MB (JSON, not binary).
- The demo must run without touching any `inside/` data at runtime.
