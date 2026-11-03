# UI plan — hackathon demo

Single-page app, two tabs: **Graphs** and **Mobility**. No server required — all data pre-loaded as static files. Launched locally with `python -m http.server` or equivalent.

---

## Global layout

- Header: project name + tab switcher (Graphs | Mobility)
- Main area: split left/right or top/bottom depending on content
- No login, no form inputs, no live computation

---

## Tab 1: Graphs

### Left panel — graph viewer

Two side-by-side force-directed graph canvases: **Real** and **Synthetic**.

- Nodes coloured by one attribute (school group, 5–6 values + grey for "none")
- Edges drawn faintly; node size proportional to degree (log scale, capped)
- Pre-loaded subsets: ~300 nodes each (a connected subgraph of the full graph — ego neighbourhood around a hub, or the largest connected component trimmed to 300)
- Controls: attribute selector (dropdown: school / hometown / location); toggles zoom/pan; a "reset view" button
- No physics simulation on load — use a pre-computed layout (e.g. force-atlas positions saved to JSON once)

### Right panel — comparison figures

Three static charts, vertically stacked:

1. **Degree distribution** — log-log histogram: Real (blue) vs Synthetic (orange) vs BTER baseline (grey)
2. **Attribute mixing** — grouped bar chart: for the selected attribute, fraction of edges that are same-group vs cross-group, real vs synthetic
3. **Real-vs-fake discriminator importance** — horizontal bar chart: feature groups (attributes / degree / structure) with their AUC drop; attributes bar annotated "fixing in v2"

Charts are static SVG or rendered once with a charting lib; no interactivity needed beyond the attribute selector updating chart 2.

---

## Tab 2: Mobility

### Left panel — map

Leaflet.js map centred on Beijing. Two layers toggled by a toggle button:

- **Real traces**: 10–15 sampled user traces (polylines, one colour per user, semi-transparent)
- **Synthetic traces**: same number of synthetic traces in a visually distinct palette

Below the map: a city selector (Beijing / Amsterdam) — switching loads a different set of pre-generated synthetic traces anchored to that city's bounding box. Real traces stay on Beijing only.

### Right panel — comparison figures

Two static charts:

1. **Outing length distribution** — histogram: Real (blue) vs Synthetic (orange)
2. **Revisit rate** — bar: Real vs Synthetic (fraction of outings that revisit a previously-visited stop cell)

Optional third (if data is easy to compute): **co-visit degree distribution** (how many unique co-visitors per user).

---

## What is NOT in the UI

- No live data loading, no API calls
- No editable parameters (ε slider etc.) — these are in the slides, not the demo
- No Adult Census tab
- No animation or video
- No authentication
