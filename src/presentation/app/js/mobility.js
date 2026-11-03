/* mobility.js — loaded as a plain <script> tag after Leaflet and Chart.js CDN */
/* globals: L, Chart */

(function () {
  'use strict';

  /* ── map ── */
  var map = L.map('map', { zoomControl: true }).setView([39.92, 116.41], 11);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 18
  }).addTo(map);

  /* ── colour palettes ── */
  var REAL_COLORS = [
    '#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f',
    '#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac',
    '#86bcb6','#e15759'
  ];
  var SYNTH_COLORS = [
    '#a8c4e0','#f9c97c','#f0a0a0','#b8dbd9','#a2d49a',
    '#f6e5a0','#d4b8d4','#ffccd4','#c8a898','#d8d4d0',
    '#c0dcd8','#f0a0a0'
  ];

  /* ── layer groups ── */
  var realLayer  = L.layerGroup().addTo(map);
  var synthLayer = L.layerGroup().addTo(map);

  /* ── helper: build layer group from GeoJSON FeatureCollection ── */
  function buildTraceLayer(geojson, palette, layerGroup) {
    layerGroup.clearLayers();
    var features = geojson.features || [];
    features.forEach(function (feature, i) {
      var coords = (feature.geometry && feature.geometry.coordinates) || [];
      var latLngs = coords.map(function (c) { return L.latLng(c[1], c[0]); });
      var color   = palette[i % palette.length];
      L.polyline(latLngs, { color: color, weight: 2, opacity: 0.7 })
        .addTo(layerGroup);
    });
  }

  /* ── toggle buttons ── */
  var btnReal  = document.getElementById('btn-real');
  var btnSynth = document.getElementById('btn-synth');
  var realVisible  = true;
  var synthVisible = true;

  btnReal.addEventListener('click', function () {
    if (realVisible) {
      map.removeLayer(realLayer);
      btnReal.classList.remove('active');
    } else {
      realLayer.addTo(map);
      btnReal.classList.add('active');
    }
    realVisible = !realVisible;
  });

  btnSynth.addEventListener('click', function () {
    if (synthVisible) {
      map.removeLayer(synthLayer);
      btnSynth.classList.remove('active');
    } else {
      synthLayer.addTo(map);
      btnSynth.classList.add('active');
    }
    synthVisible = !synthVisible;
  });

  /* ── resize fix: map is hidden when page loads ── */
  window.addEventListener('tab-changed', function () {
    if (document.getElementById('tab-mobility').classList.contains('active')) {
      map.invalidateSize();
    }
  });

  /* ── Chart.js global config (guard against graphs.js already setting it) ── */
  if (Chart.defaults.color !== '#8b949e') {
    Chart.defaults.color       = '#8b949e';
    Chart.defaults.borderColor = '#21262d';
  }

  /* ── charts ── */
  var chartOutingLength = null;
  var chartRevisit      = null;

  function renderOutingLengthChart(stats) {
    var ctx  = document.getElementById('chart-outing-length').getContext('2d');
    var data = stats.outing_length_km;
    if (chartOutingLength) { chartOutingLength.destroy(); }
    chartOutingLength = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.bins,
        datasets: [
          {
            label: 'Real',
            data: data.real,
            backgroundColor: '#4e79a7'
          },
          {
            label: 'Synthetic',
            data: data.synthetic,
            backgroundColor: '#f28e2b'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Share of outings' }
          }
        }
      }
    });
  }

  function renderRevisitChart(stats) {
    var ctx  = document.getElementById('chart-revisit').getContext('2d');
    var data = stats.revisit_rate;
    if (chartRevisit) { chartRevisit.destroy(); }
    chartRevisit = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Real', 'Synthetic'],
        datasets: [
          {
            label: 'Revisit rate',
            data: [data.real, data.synthetic],
            backgroundColor: ['#4e79a7', '#f28e2b']
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          subtitle: {
            display: true,
            text: 'Synthetic: placeholder (final run pending)'
          }
        },
        scales: {
          x: {
            min: 0,
            max: 1,
            title: {
              display: true,
              text: 'Fraction of outings that revisit a stop'
            }
          }
        }
      }
    });
  }

  /* ── fetch everything in parallel ── */
  Promise.all([
    fetch('data/traces_real_beijing.geojson').then(function (r) {
      if (!r.ok) { throw new Error('traces_real fetch failed: ' + r.status); }
      return r.json();
    }),
    fetch('data/traces_synth_beijing.geojson').then(function (r) {
      if (!r.ok) { throw new Error('traces_synth fetch failed: ' + r.status); }
      return r.json();
    }),
    fetch('data/mobility_stats.json').then(function (r) {
      if (!r.ok) { throw new Error('mobility_stats fetch failed: ' + r.status); }
      return r.json();
    })
  ]).then(function (results) {
    var realGeoJSON  = results[0];
    var synthGeoJSON = results[1];
    var stats        = results[2];

    buildTraceLayer(realGeoJSON,  REAL_COLORS,  realLayer);
    buildTraceLayer(synthGeoJSON, SYNTH_COLORS, synthLayer);

    renderOutingLengthChart(stats);
    renderRevisitChart(stats);
  }).catch(function (err) {
    console.error('[mobility.js]', err);
  });

}());
