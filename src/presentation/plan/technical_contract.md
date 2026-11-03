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

## Data — what you provide

Drop your files into `src/presentation/app/data/` when the final algorithm runs are done. The app reads them directly; no build step or conversion script needed. Required formats are listed below.

### Graph data — required export format

**Preferred: GraphML** (`.graphml`). Sigma.js v2 has a built-in GraphML parser. Export from networkx with `nx.write_graphml(G, path)`. The app reads two files:

- `graph_real_subset.graphml` — a subgraph of ~300 nodes (e.g. ego neighbourhood of the largest hub, trimmed to 300 by degree). Each node must carry attributes: `degree` (int), `school`, `hometown`, `location` (string category labels or integer strings). Include pre-computed layout positions as node attributes `x` (float) and `y` (float) — run `spring_layout` or `forceatlas2` once and bake the positions in. Without positions Sigma falls back to a random layout which is slow and ugly.
- `graph_synthetic_subset.graphml` — same structure, same node-attribute keys, comparable size subset from the synthetic graph.

**Fallback: JSON** using the schema below if GraphML export is inconvenient.

### Graph stats — `graph_stats.json`

Compute from your final eval outputs and drop this file:

```json
{
  "degree_hist": {
    "bins": [1, 2, 4, 8, 16, 32, 64, 128],
    "real":      [120, 95, 70, 40, 20, 8, 3, 1],
    "synthetic": [115, 90, 68, 38, 18, 7, 3, 1],
    "bter":      [100, 88, 65, 35, 16, 6, 2, 1]
  },
  "mixing": {
    "school":   { "real_same": 0.59, "synth_same": 0.42 },
    "hometown": { "real_same": 0.55, "synth_same": 0.38 },
    "location": { "real_same": 0.52, "synth_same": 0.41 }
  },
  "discriminator": {
    "attributes": 0.43,
    "degree":     0.18,
    "structure":  0.05
  }
}
```

Source: `snap/eval/results_v1_1/` (coupling_reference.csv for mixing, discriminator_importance_v1_1.csv for the last block).

### Mobility data — required export format

- `traces_real_beijing.geojson` — GeoJSON FeatureCollection of LineString features, one per user, property `user_id`. 10–15 users, up to 10 outings each. **Clip coordinates to H3-6 cell resolution (truncate to ~3 decimal places) — do not export raw GPS precision from the real dataset.**
- `traces_synth_beijing.geojson` — same format, synthetic users.
- `traces_synth_amsterdam.geojson` — synthetic users anchored to Amsterdam, if available. If not, omit and the city selector will be hidden.
- `mobility_stats.json`:

```json
{
  "outing_length_km": {
    "bins":      ["0–5", "5–15", "15–30", "30+"],
    "real":      [0.12, 0.35, 0.31, 0.22],
    "synthetic": [0.18, 0.40, 0.28, 0.14]
  },
  "revisit_rate": {
    "real":      0.59,
    "synthetic": 0.43
  }
}
```

---

## Graph JSON fallback schema

Only needed if GraphML export is not available:

```json
{
  "nodes": [
    { "id": "42", "x": 0.3, "y": -0.7, "degree": 15,
      "school": "alumni_net_0", "hometown": "city_2", "location": "loc_1" }
  ],
  "edges": [
    { "source": "42", "target": "17" }
  ]
}
```

A top-level `"label_map"` key is optional: maps raw attribute values to display names.

---

## Constraints

- No network calls during the demo — all data is local static files.
- Graph layout pre-computed (positions in JSON) — no physics on page load.
- OSM tile layer requires internet; if the venue has no wifi, swap for a static map image background in Leaflet.
- Real GPS traces: clip to the H3-6 cell covering Beijing central (not sub-city precision) before export — do not export raw GPS coordinates from the real dataset into a browser-accessible file.
- Total `data/` directory should be under 5 MB (JSON, not binary).
- The demo must run without touching any `inside/` data at runtime.
