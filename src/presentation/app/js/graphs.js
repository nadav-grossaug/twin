(function () {
  'use strict';

  // ── Global Chart.js defaults ───────────────────────────────────────────────
  Chart.defaults.color = '#8b949e';
  Chart.defaults.borderColor = '#21262d';

  var PALETTE = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948'];
  var GREY = '#3a3f4b';

  // ── State ──────────────────────────────────────────────────────────────────
  var realGraphData = null;
  var synthGraphData = null;
  var statsData = null;
  var realG = null;
  var synthG = null;
  var realRenderer = null;
  var synthRenderer = null;
  var colorMap = {};
  var mixingChart = null;
  var currentAttr = 'school';

  var realContainer = document.getElementById('graph-real');
  var synthContainer = document.getElementById('graph-synth');

  // ── Loading / error helpers ────────────────────────────────────────────────
  function showLoading() {
    var style = 'color:#8b949e;padding:20px;display:flex;align-items:center;justify-content:center;height:100%';
    realContainer.innerHTML = '<div style="' + style + '">Loading…</div>';
    synthContainer.innerHTML = '<div style="' + style + '">Loading…</div>';
  }

  function showError() {
    var style = 'color:#e15759;padding:20px;display:flex;align-items:center;justify-content:center;height:100%';
    realContainer.innerHTML = '<div style="' + style + '">Failed to load data</div>';
    synthContainer.innerHTML = '<div style="' + style + '">Failed to load data</div>';
  }

  // ── Color-map builder ──────────────────────────────────────────────────────
  // Counts frequency of each attribute value across BOTH graphs combined.
  // Top 6 non-"none" values get PALETTE colors; everything else maps to GREY.
  // "none" always maps to GREY.
  function buildColorMap(attr) {
    var freq = {};
    [realGraphData, synthGraphData].forEach(function (gd) {
      gd.nodes.forEach(function (n) {
        var v = (n[attr] != null) ? n[attr] : 'none';
        freq[v] = (freq[v] || 0) + 1;
      });
    });

    var sorted = Object.keys(freq)
      .filter(function (v) { return v !== 'none'; })
      .sort(function (a, b) { return freq[b] - freq[a]; });

    var map = { none: GREY };
    sorted.slice(0, 6).forEach(function (v, i) {
      map[v] = PALETTE[i];
    });

    return map;
  }

  function nodeColor(val) {
    return (colorMap[val] != null) ? colorMap[val] : GREY;
  }

  // ── Build graphology + Sigma for one container ─────────────────────────────
  function buildGraph(gd, container) {
    container.innerHTML = '';   // clear loading indicator

    var g = new graphology.Graph({ multi: false, type: 'undirected' });

    gd.nodes.forEach(function (node) {
      g.addNode(node.id, {
        x: node.x,
        y: node.y,
        size: Math.max(2, Math.log1p(node.degree) * 2),
        color: nodeColor(node[currentAttr]),
        label: node.id,
        // cache all three attribute values for fast recoloring
        _school: node.school,
        _hometown: node.hometown,
        _location: node.location,
      });
    });

    gd.edges.forEach(function (edge) {
      try {
        g.addEdge(edge.source, edge.target);
      } catch (e) {
        // ignore duplicate or self-loop edges
      }
    });

    var renderer = new Sigma(g, container, {
      renderEdgeLabels: false,
      defaultEdgeColor: '#2a3040',
      labelColor: { color: '#8b949e' },
      labelSize: 9,
      minCameraRatio: 0.05,
      maxCameraRatio: 8,
    });

    // Defer refresh so the flex container has its final pixel size
    setTimeout(function () {
      renderer.refresh();
      renderer.getCamera().animatedReset();
    }, 50);

    return { g: g, renderer: renderer };
  }

  // ── Update node colors on both live graphology graphs ──────────────────────
  function recolorGraphs(attr) {
    if (!realG || !synthG) return;
    var key = '_' + attr;
    realG.forEachNode(function (id, attrs) {
      realG.setNodeAttribute(id, 'color', nodeColor(attrs[key]));
    });
    synthG.forEachNode(function (id, attrs) {
      synthG.setNodeAttribute(id, 'color', nodeColor(attrs[key]));
    });
    realRenderer.refresh();
    synthRenderer.refresh();
  }

  // ── Legend ─────────────────────────────────────────────────────────────────
  function updateLegend() {
    var legend = document.getElementById('graph-legend');
    legend.innerHTML = '';

    // Named top-6 entries
    Object.keys(colorMap).forEach(function (v) {
      if (v === 'none') return;   // handled below as combined "other / none"
      var item = document.createElement('div');
      item.className = 'legend-item';
      var dot = document.createElement('span');
      dot.className = 'legend-dot';
      dot.style.background = colorMap[v];
      item.appendChild(dot);
      item.appendChild(document.createTextNode(v));
      legend.appendChild(item);
    });

    // Combined "other / none" bucket
    var item = document.createElement('div');
    item.className = 'legend-item';
    var dot = document.createElement('span');
    dot.className = 'legend-dot';
    dot.style.background = GREY;
    item.appendChild(dot);
    item.appendChild(document.createTextNode('other / none'));
    legend.appendChild(item);
  }

  // ── Chart 1: degree distribution ──────────────────────────────────────────
  function buildDegreeChart() {
    var dh = statsData.degree_hist;
    new Chart(document.getElementById('chart-degree'), {
      type: 'bar',
      data: {
        labels: dh.bins,
        datasets: [
          { label: 'Real',          data: dh.real,      backgroundColor: '#4e79a7' },
          { label: 'Synthetic',     data: dh.synthetic, backgroundColor: '#f28e2b' },
          { label: 'BTER baseline', data: dh.bter,      backgroundColor: '#5a6473' },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: {
          x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
          y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
        },
      },
    });
  }

  // ── Chart 2: attribute mixing ──────────────────────────────────────────────
  function buildMixingChart() {
    var mx = statsData.mixing[currentAttr];
    mixingChart = new Chart(document.getElementById('chart-mixing'), {
      type: 'bar',
      data: {
        labels: ['Real', 'Synthetic'],
        datasets: [{
          label: 'Same-group edge fraction',
          data: [mx.real_same, mx.synth_same],
          backgroundColor: ['#4e79a7', '#f28e2b'],
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: '#8b949e' },
            grid:  { color: '#21262d' },
          },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: 'Same-group edge fraction', color: '#8b949e' },
            ticks: { color: '#8b949e' },
            grid:  { color: '#21262d' },
          },
        },
      },
    });
  }

  function updateMixingChart() {
    if (!mixingChart || !statsData) return;
    var mx = statsData.mixing[currentAttr];
    mixingChart.data.datasets[0].data = [mx.real_same, mx.synth_same];
    mixingChart.update();
  }

  // ── Chart 3: real-vs-fake discriminator ───────────────────────────────────
  function buildDiscrimChart() {
    var disc = statsData.discriminator;
    new Chart(document.getElementById('chart-discrim'), {
      type: 'bar',
      data: {
        labels: ['Attributes', 'Degree', 'Structure'],
        datasets: [{
          label: 'AUC drop (feature importance)',
          data: [disc.attributes, disc.degree, disc.structure],
          backgroundColor: ['#e15759', '#f28e2b', '#5a6473'],
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: function (ctx) {
                return ctx.dataIndex === 0 ? 'fixing in v2' : '';
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'AUC drop (feature importance)', color: '#8b949e' },
            ticks: { color: '#8b949e' },
            grid:  { color: '#21262d' },
          },
          y: {
            ticks: { color: '#8b949e' },
            grid:  { color: '#21262d' },
          },
        },
      },
    });
  }

  // ── Full init after all data is loaded ────────────────────────────────────
  function init(realData, synthData, stats) {
    realGraphData = realData;
    synthGraphData = synthData;
    statsData = stats;

    currentAttr = document.getElementById('attr-select').value || 'school';
    colorMap = buildColorMap(currentAttr);

    var rr = buildGraph(realData, realContainer);
    realG = rr.g;
    realRenderer = rr.renderer;

    var sr = buildGraph(synthData, synthContainer);
    synthG = sr.g;
    synthRenderer = sr.renderer;

    updateLegend();
    buildDegreeChart();
    buildMixingChart();
    buildDiscrimChart();

    // Wire up the attribute selector
    document.getElementById('attr-select').addEventListener('change', function (e) {
      currentAttr = e.target.value;
      colorMap = buildColorMap(currentAttr);
      recolorGraphs(currentAttr);
      updateLegend();
      updateMixingChart();
      document.getElementById('mixing-attr-label').textContent = currentAttr;
    });
  }

  // ── Boot ───────────────────────────────────────────────────────────────────
  showLoading();

  Promise.all([
    fetch('data/graph_real_subset.json').then(function (r) { return r.json(); }),
    fetch('data/graph_synthetic_subset.json').then(function (r) { return r.json(); }),
    fetch('data/graph_stats.json').then(function (r) { return r.json(); }),
  ]).then(function (results) {
    init(results[0], results[1], results[2]);
  }).catch(function (err) {
    console.error('graphs.js: failed to load data', err);
    showError();
  });

}());
