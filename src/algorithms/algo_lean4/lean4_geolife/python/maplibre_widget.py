"""MapLibre GL JS map as an anywidget, shared by the enclave and remote notebooks.

GeoJSON layers go in (`layers` dict keyed by source id); the clicked stay/region
comes out (`clicked`). Supported sources: density, tracks, regions, flows, gaps,
steps, stays, fixes (drawn in that order; fixes = raw GPS points, steps = the
line between two consecutive fixes, gaps = recording gaps between them).
Only sources present in `layers` are drawn.

Network: the GeoJSON is passed to the browser in-process. A basemap fetches tiles
for the viewed area (`basemap="none"` makes no requests). The library itself loads
from `lib_url`. Override it with GEOLIFE_MAPLIBRE_URL to serve a vendored copy.
"""

from __future__ import annotations

import os

import anywidget
import traitlets

MAPLIBRE_URL = os.environ.get("GEOLIFE_MAPLIBRE_URL", "https://unpkg.com/maplibre-gl@6.11.2/dist/maplibre-gl")
BASEMAPS = {"OpenFreeMap Liberty (tiles from tiles.openfreemap.org)": "https://tiles.openfreemap.org/styles/liberty",
            "OpenFreeMap Dark (tiles from tiles.openfreemap.org)": "https://tiles.openfreemap.org/styles/dark",
            "none (no network)": "none"}


def default_basemap_label(env_default: str = "liberty") -> str:
    """Basemap option chosen by GEOLIFE_BASEMAP (liberty | dark | none | style URL)."""
    env = os.environ.get("GEOLIFE_BASEMAP", env_default)
    return next((k for k, v in BASEMAPS.items() if env in (v, k) or (env != "none" and env in v)), "none (no network)")


def feature_collection(features: list) -> dict:
    return {"type": "FeatureCollection", "features": features}


class MapLibreMap(anywidget.AnyWidget):
    """Minimal MapLibre GL JS widget: GeoJSON layers in, clicked feature out."""

    _esm = r"""
    const ROLE = ["match", ["get", "role"], "r1", "#e45756", "r2", "#4c78a8", "r3_5", "#54a24b",
                  "other", "#54a24b", "r6_plus", "#b279a2", "#9d9d9d"];
    const MODE = ["match", ["get", "mode"], "walk", "#f58518", "bike", "#54a24b", "bus", "#4c78a8",
                  "car", "#e45756", "taxi", "#eeca3b", "subway", "#b279a2", "train", "#9d755d",
                  "airplane", "#72b7b2", "#bab0ac"];
    const SPEED = ["interpolate", ["linear"], ["get", "speed_kmh"], 0, "#2c7bb6", 5, "#00a6ca", 15, "#90eb9d",
                   30, "#f9d057", 60, "#d7191c"];
    const NO_BASEMAP = {version: 8, sources: {}, layers: [
      {id: "bg", type: "background", paint: {"background-color": "#14171c"}}]};
    // metres -> pixels at Beijing's latitude (Web Mercator, 512 px tiles)
    // (zoom must be the top-level interpolate input, so clamping goes inside the stop outputs)
    const metres = (m, minPx) => ["interpolate", ["exponential", 2], ["zoom"],
                                  0, ["max", minPx, ["/", m, 120050]], 22, ["max", minPx, ["*", m, 4194304 / 120050]]];
    const LAYERS = [
      {id: "density", source: "density", type: "heatmap", maxzoom: 15, paint: {
        "heatmap-weight": ["interpolate", ["linear"], ["ln", ["+", 1, ["get", "n"]]], 0, 0, 12, 1],
        "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 4, 0.15, 10, 0.4, 14, 0.6],
        "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 4, 2, 10, 6, 14, 14],
        // sparse areas stay transparent so the basemap and the user's layers remain readable
        "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"],
          0, "rgba(0,0,0,0)", 0.25, "rgba(49,54,149,0.25)", 0.5, "rgba(69,117,180,0.5)",
          0.7, "rgba(254,224,144,0.7)", 0.9, "rgba(244,109,67,0.85)", 1, "rgba(165,0,38,0.9)"],
        "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 12, 0.7, 15, 0]}},
      {id: "tracks", source: "tracks", type: "line", layout: {"line-cap": "round", "line-join": "round"},
       paint: {"line-color": MODE, "line-width": ["interpolate", ["linear"], ["zoom"], 9, 1, 16, 3],
               "line-opacity": 0.8}},
      {id: "regions", source: "regions", type: "circle", paint: {
        "circle-radius": metres(["max", 75, ["get", "radius_m"]], 4),
        "circle-color": ROLE, "circle-opacity": 0.18, "circle-stroke-color": ROLE, "circle-stroke-width": 1.5}},
      {id: "flows", source: "flows", type: "line", layout: {"line-cap": "round"}, paint: {
        "line-color": ["case", ["get", "motif"], "#ff9f1c", "#9ecae1"],
        "line-width": ["interpolate", ["linear"], ["get", "n"], 1, 1, 20, 8], "line-opacity": 0.85}},
      {id: "gaps", source: "gaps", type: "line", paint: {
        "line-color": "#9d9d9d", "line-width": 1.5, "line-dasharray": [2, 2], "line-opacity": 0.9}},
      {id: "steps", source: "steps", type: "line", layout: {"line-cap": "round"}, paint: {
        "line-color": SPEED, "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1, 16, 3.5], "line-opacity": 0.9}},
      {id: "stays", source: "stays", type: "circle", paint: {
        "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "dwell_h"]], 0, 3, 5, 12],
        "circle-color": ROLE, "circle-opacity": 0.85,
        "circle-stroke-color": ["case", ["get", "censored"], "#ffffff", "#14171c"],
        "circle-stroke-width": ["case", ["get", "censored"], 2, 0.5]}},
      {id: "fixes", source: "fixes", type: "circle", paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 10, 1.5, 17, 5], "circle-color": MODE,
        "circle-stroke-color": "#14171c", "circle-stroke-width": 0.6}},
    ];
    // click priority: the most specific feature under the cursor wins
    const CLICKABLE = ["fixes", "stays", "steps", "gaps", "regions"];

    function table(props) {
      const t = document.createElement("table");
      t.style.cssText = "font: 12px system-ui; color: #111; border-collapse: collapse";
      for (const [k, v] of Object.entries(props)) {
        const tr = t.insertRow();
        const a = tr.insertCell(); a.textContent = k; a.style.cssText = "padding: 1px 8px 1px 0; color: #666";
        const b = tr.insertCell(); b.textContent = typeof v === "number" ? +v.toFixed(3) : String(v);
      }
      return t;
    }

    async function render({ model, el }) {
      const url = model.get("lib_url");
      const lib = await import(`${url}.mjs`);
      const css = document.createElement("link");
      css.rel = "stylesheet"; css.href = `${url}.css`;
      el.appendChild(css);
      const box = document.createElement("div");
      box.style.cssText = `width: 100%; height: ${model.get("height")}px; border-radius: 6px; overflow: hidden`;
      el.appendChild(box);
      const style = () => model.get("basemap") === "none" ? NO_BASEMAP : model.get("basemap");
      const map = new lib.Map({container: box, style: style(), bounds: model.get("bounds"),
                               fitBoundsOptions: {padding: 40}, attributionControl: {compact: true}});
      // the notebook may size the container after creation: follow it, and fit the bounds once loaded
      const ro = new ResizeObserver(() => map.resize());
      ro.observe(box);
      map.once("load", () => { map.resize(); map.fitBounds(model.get("bounds"), {padding: 40, duration: 0}); });
      map.addControl(new lib.NavigationControl(), "top-right");
      map.addControl(new lib.ScaleControl({unit: "metric"}), "bottom-left");

      const addData = () => {
        const data = model.get("layers"), visible = new Set(model.get("visible"));
        for (const L of LAYERS) {
          if (!data[L.source]) continue;
          if (!map.getSource(L.source)) map.addSource(L.source, {type: "geojson", data: data[L.source]});
          if (!map.getLayer(L.id)) map.addLayer({...L, layout: {...(L.layout || {}),
            visibility: visible.has(L.id) ? "visible" : "none"}});
        }
      };
      map.on("style.load", addData);
      model.on("change:visible", () => {
        const visible = new Set(model.get("visible"));
        for (const L of LAYERS) if (map.getLayer(L.id))
          map.setLayoutProperty(L.id, "visibility", visible.has(L.id) ? "visible" : "none");
      });
      model.on("change:basemap", () => map.setStyle(style()));
      // one handler: fixes sit on steps, stays inside region circles, so prefer the most specific hit
      const clickable = () => CLICKABLE.filter((id) => map.getLayer(id) &&
                                 map.getLayoutProperty(id, "visibility") !== "none");
      map.on("click", (e) => {
        const box = [[e.point.x - 3, e.point.y - 3], [e.point.x + 3, e.point.y + 3]];   // thin lines need a margin
        const hits = map.queryRenderedFeatures(box, {layers: clickable()});
        if (!hits.length) return;
        const f = CLICKABLE.map((id) => hits.find((h) => h.layer.id === id)).find(Boolean);
        new lib.Popup({maxWidth: "320px"}).setLngLat(e.lngLat).setDOMContent(table(f.properties)).addTo(map);
        model.set("clicked", {layer: f.layer.id, ...f.properties}); model.save_changes();
      });
      map.on("mousemove", (e) => {
        map.getCanvas().style.cursor = map.queryRenderedFeatures(e.point, {layers: clickable()}).length ? "pointer" : "";
      });
      return () => { ro.disconnect(); map.remove(); };
    }
    export default { render };
    """
    lib_url = traitlets.Unicode(MAPLIBRE_URL).tag(sync=True)
    basemap = traitlets.Unicode("none").tag(sync=True)
    bounds = traitlets.List([116.1, 39.7, 116.7, 40.1]).tag(sync=True)
    height = traitlets.Int(620).tag(sync=True)
    layers = traitlets.Dict().tag(sync=True)
    visible = traitlets.List().tag(sync=True)
    clicked = traitlets.Dict().tag(sync=True)
