/* mobility.js — loaded as a plain <script> tag after Leaflet and Chart.js CDN */
/* globals: L, Chart */

(function () {
  'use strict';

  /* ── Chart.js global config (guard against graphs.js already setting it) ── */
  if (Chart.defaults.color !== '#8b949e') {
    Chart.defaults.color       = '#8b949e';
    Chart.defaults.borderColor = '#21262d';
  }

  /* ── init two maps ── */
  function makeMap(id) {
    return L.map(id, { zoomControl: true, attributionControl: false })
      .setView([39.92, 116.41], 10);
  }

  var mapReal  = makeMap('map-real');
  var mapSynth = makeMap('map-synth');

  [mapReal, mapSynth].forEach(function (m) {
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18
    }).addTo(m);
  });

  var HEAT_OPTS = {
    radius: 12, blur: 10, minOpacity: 0.35,
    gradient: { 0.2: '#3182bd', 0.5: '#74c476', 0.8: '#fd8d3c', 1.0: '#de2d26' }
  };

  /* ── extract points and bounds from GeoJSON, don't touch the map yet ── */
  function parseGeoJSON(geojson) {
    var points = [], allLat = [], allLng = [];
    (geojson.features || []).forEach(function (f) {
      (f.geometry && f.geometry.coordinates || []).forEach(function (c) {
        allLat.push(c[1]); allLng.push(c[0]);
        points.push([c[1], c[0], 1]);
      });
    });
    var bounds = allLat.length ? L.latLngBounds(
      L.latLng(Math.min.apply(null, allLat), Math.min.apply(null, allLng)),
      L.latLng(Math.max.apply(null, allLat), Math.max.apply(null, allLng))
    ) : null;
    return { points: points, bounds: bounds };
  }

  /* ── stored parsed data — heat layers built after invalidateSize ── */
  var parsed = { real: null, synth: null };
  var heatLayers = { real: null, synth: null };

  function attachHeat(map, key) {
    if (!parsed[key]) return;
    if (heatLayers[key]) { map.removeLayer(heatLayers[key]); }
    heatLayers[key] = L.heatLayer(parsed[key].points, HEAT_OPTS).addTo(map);
  }

  function applyBounds() {
    mapReal.invalidateSize();
    mapSynth.invalidateSize();
    /* build heat layers now that maps have real pixel dimensions */
    attachHeat(mapReal,  'real');
    attachHeat(mapSynth, 'synth');
    if (parsed.real  && parsed.real.bounds)  mapReal.fitBounds(parsed.real.bounds,   { padding: [20, 20] });
    if (parsed.synth && parsed.synth.bounds) mapSynth.fitBounds(parsed.synth.bounds, { padding: [20, 20] });
  }

  window.addEventListener('tab-changed', function () {
    if (document.getElementById('tab-mobility').classList.contains('active')) {
      setTimeout(applyBounds, 80);
    }
  });

  /* ── chart helpers ── */
  var BLUE   = '#4e79a7';
  var ORANGE = '#f28e2b';
  var GREY   = '#5a6473';
  var RED    = '#e15759';

  var chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'top' } },
    scales: {
      x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
    }
  };

  function deepMerge(base, extra) {
    var out = JSON.parse(JSON.stringify(base));
    Object.keys(extra).forEach(function (k) {
      if (extra[k] && typeof extra[k] === 'object' && !Array.isArray(extra[k])) {
        out[k] = deepMerge(out[k] || {}, extra[k]);
      } else {
        out[k] = extra[k];
      }
    });
    return out;
  }

  function renderOutingLengthChart(stats) {
    var data = stats.outing_length_km;
    new Chart(document.getElementById('chart-outing-length').getContext('2d'), {
      type: 'bar',
      data: {
        labels: data.bins,
        datasets: [
          { label: 'Real',      data: data.real,      backgroundColor: BLUE },
          { label: 'Synthetic', data: data.synthetic, backgroundColor: ORANGE }
        ]
      },
      options: deepMerge(chartOpts, {
        scales: { y: { title: { display: true, text: 'Share of outings', color: '#8b949e' } } }
      })
    });
  }

  function renderRevisitChart(stats) {
    var data = stats.revisit_rate;
    new Chart(document.getElementById('chart-revisit').getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['Real', 'Synthetic'],
        datasets: [{
          label: 'Revisit rate',
          data: [data.real, data.synthetic],
          backgroundColor: [BLUE, ORANGE]
        }]
      },
      options: deepMerge(chartOpts, {
        plugins: {
          legend: { display: false },
          subtitle: { display: true, text: 'Fraction of outings returning to a prior stop', color: '#8b949e' }
        },
        scales: { y: { min: 0, max: 1 } }
      })
    });
  }

  function renderMetricsChart(stats) {
    /* normalised comparison of key scalars, real=1 baseline */
    var rv  = stats.revisit_rate;
    var ol  = stats.outing_length_km;
    /* mean outing length (weighted) */
    var mids  = [2.5, 10, 22.5, 37.5];
    function wmean(shares) {
      return mids.reduce(function (s, m, i) { return s + m * shares[i]; }, 0);
    }
    var realMean  = wmean(ol.real);
    var synthMean = wmean(ol.synthetic);

    new Chart(document.getElementById('chart-metrics').getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['Revisit rate', 'Mean outing length (km)'],
        datasets: [
          { label: 'Real',      data: [rv.real,  realMean],  backgroundColor: BLUE },
          { label: 'Synthetic', data: [rv.synthetic, synthMean], backgroundColor: ORANGE }
        ]
      },
      options: deepMerge(chartOpts, {
        plugins: { legend: { position: 'top' } }
      })
    });
  }

  function renderDiscrimChart() {
    /* GeoLife discriminator — from docs/realfake/geolife.md section 5:
       gbm AUC 0.9996 with map / 0.9995 without;
       feature group importance (approx from write-up) */
    new Chart(document.getElementById('chart-discrim-mobility').getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['Step timing', 'Kinematics', 'Map features', 'Size'],
        datasets: [{
          label: 'AUC drop (feature importance)',
          data: [0.41, 0.28, 0.08, 0.03],
          backgroundColor: [RED, ORANGE, BLUE, GREY]
        }]
      },
      options: deepMerge(chartOpts, {
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          subtitle: { display: true, text: 'Real-vs-fake classifier · GeoLife · AUC 0.9996', color: '#8b949e' }
        },
        scales: {
          x: { title: { display: true, text: 'AUC drop', color: '#8b949e' } }
        }
      })
    });
  }

  /* ── fetch everything and wire up ── */
  Promise.all([
    fetch('data/traces_real_beijing.geojson').then(function (r) {
      if (!r.ok) throw new Error('traces_real: ' + r.status);
      return r.json();
    }),
    fetch('data/traces_synth_beijing.geojson').then(function (r) {
      if (!r.ok) throw new Error('traces_synth: ' + r.status);
      return r.json();
    }),
    fetch('data/mobility_stats.json').then(function (r) {
      if (!r.ok) throw new Error('mobility_stats: ' + r.status);
      return r.json();
    })
  ]).then(function (results) {
    var realGeoJSON  = results[0];
    var synthGeoJSON = results[1];
    var stats        = results[2];

    /* parse points+bounds; heat layers are built on first tab-changed */
    try {
      parsed.real  = parseGeoJSON(realGeoJSON);
      parsed.synth = parseGeoJSON(synthGeoJSON);
    } catch (e) {
      console.error('[mobility.js] parse error:', e);
    }
    /* if already on the mobility tab when data loads, render immediately */
    if (document.getElementById('tab-mobility').classList.contains('active')) {
      setTimeout(applyBounds, 80);
    }

    /* charts — each isolated so one failure doesn't kill the rest */
    [
      function () { renderOutingLengthChart(stats); },
      function () { renderRevisitChart(stats); },
      function () { renderMetricsChart(stats); },
      function () { renderDiscrimChart(); }
    ].forEach(function (fn) {
      try { fn(); } catch (e) { console.error('[mobility.js] chart error:', e); }
    });

  }).catch(function (err) {
    console.error('[mobility.js] fetch error:', err);
  });

}());
