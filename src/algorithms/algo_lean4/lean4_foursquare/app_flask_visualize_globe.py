#!/usr/bin/env python3
"""Animated globe of country-to-country user transfers.

    python app_flask_visualize_globe.py --left country_transfer.tsv [--right synth/travel_synth.tsv] [--port 5055]

With --right, two globes are shown side by side (left = --left, right = --right),
sharing one clock and one camera so both play in tandem.

Input TSV columns (as produced by the pipeline): userid, dt, prev_country,
next_country, prev_last_lat/long, next_first_lat/long, ...
Each row is one transfer of a user from prev_country to next_country at dt.
Served page uses MapLibre GL JS (globe projection) to draw dashed arcs and
moving dots, with a time slider and play/stop/back/forward controls.
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from flask import Flask, jsonify


def load(path, ref=None):
    pts = defaultdict(lambda: ([], []))
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                t = int(datetime.strptime(r["dt"], "%Y-%m-%d %H:%M:%S")
                        .replace(tzinfo=timezone.utc).timestamp())
                a, b = r["prev_country"], r["next_country"]
                if not a or not b or a == b:
                    continue
                for c, la, lo in ((a, r["prev_last_lat"], r["prev_last_long"]),
                                  (b, r["next_first_lat"], r["next_first_long"])):
                    pts[c][0].append(float(la))
                    pts[c][1].append(float(lo))
                rows.append((t, a, b))
            except (KeyError, ValueError):
                continue  # skip malformed rows
    if not rows:
        raise SystemExit(f"no usable transfer rows in {path}")
    # median is robust to outlier venues; antimeridian countries (e.g. US/FJ) accepted as-is
    cent = {c: [statistics.median(v[0]), statistics.median(v[1])] for c, v in pts.items()}
    for c in cent:  # same country -> same anchor on both globes (ref = first file)
        if ref and c in ref:
            cent[c] = ref[c]
    codes = sorted(cent)
    idx = {c: i for i, c in enumerate(codes)}
    rows.sort()
    return {
        "countries": [{"code": c, "lat": cent[c][0], "lon": cent[c][1]} for c in codes],
        "t": [r[0] for r in rows],
        "a": [idx[r[1]] for r in rows],
        "b": [idx[r[2]] for r in rows],
    }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Country transfers globe</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.6.0/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@5.6.0/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@9.1.9/dist.min.js"></script>
<style>
 html,body{margin:0;height:100%;background:#050912;color:#dfe8f5;font:14px system-ui,sans-serif}
 #panels{position:absolute;inset:0;display:flex;gap:2px}
 .panel{position:relative;flex:1;min-width:0}
 .map{position:absolute;inset:0}
 .hud{position:absolute;top:12px;left:12px;z-index:2;background:#0b142455;text-shadow:0 1px 3px #000;padding:10px 14px;border-radius:8px;
      backdrop-filter:blur(4px);max-width:300px}
 .name{font-size:12px;opacity:.7;margin-bottom:2px;word-break:break-all}
 .date{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
 .stats{opacity:.8;margin-top:4px;line-height:1.5}
 .top{margin-top:6px;font-size:12px;opacity:.85}
 #bar{position:absolute;left:12px;right:12px;bottom:12px;background:#0b1424dd;padding:10px 14px;
      border-radius:8px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 #bar button{background:#1b2c4a;color:#dfe8f5;border:1px solid #34527f;border-radius:6px;
      padding:6px 12px;font-size:16px;cursor:pointer;min-width:42px}
 #bar button:hover{background:#27406a}
 #bar button.on{background:#2f6fd0}
 #slider{flex:1;min-width:200px}
 label{font-size:12px;opacity:.85;display:flex;gap:6px;align-items:center}
 select,input[type=range]{accent-color:#4da3ff}
 select{background:#1b2c4a;color:#dfe8f5;border:1px solid #34527f;border-radius:4px;padding:3px}
 #help{position:absolute;top:12px;right:12px;background:#0b1424cc;padding:8px 12px;border-radius:8px;
      font-size:12px;line-height:1.6}
 kbd{background:#1b2c4a;border:1px solid #34527f;border-radius:3px;padding:0 5px}
</style></head><body>
<div id="panels"></div>
<div id="help"><kbd>Space</kbd> play/pause &nbsp; <kbd>S</kbd> stop (rewind)<br>
 <kbd>←</kbd> back &nbsp; <kbd>→</kbd> forward (<kbd>Shift</kbd> = ×4)<br>
 <kbd>1</kbd>–<kbd>9</kbd> speed &nbsp; <kbd>Home</kbd>/<kbd>End</kbd></div>
<div id="bar">
 <button id="bBack" title="Back (←)">⏮</button>
 <button id="bStop" title="Stop (S)">⏹</button>
 <button id="bPlay" title="Play/Pause (Space)">▶</button>
 <button id="bFwd" title="Forward (→)">⏭</button>
 <input id="slider" type="range" min="0" max="10000" value="0">
 <label>speed <select id="speed"></select></label>
 <label>fade <select id="trail"></select></label>
 <label>step <select id="step"></select></label>
</div>
<script>
const DAY = 86400;
let ARCS = true, T0, T1, now, playing = false, last = 0, panels = [], syncing = false;
const speeds = [["1 day/s",1],["3 days/s",3],["1 week/s",7],["2 weeks/s",14],["1 month/s",30],["2 months/s",60],["3 months/s",90],["6 months/s",180],["1 year/s",365]];
// fade length in REAL seconds (scaled by the playback speed into simulated days)
const trails = [["0.5 s",0.5],["1 s",1],["2 s",2],["5 s",5],["10 s",10]];
const steps  = [["1 day",1],["1 week",7],["1 month",30]];
function fill(id, arr, def){const s=document.getElementById(id);
  arr.forEach(([l,v],i)=>{const o=document.createElement('option');o.textContent=l;o.value=v;if(i===def)o.selected=true;s.appendChild(o)});}
fill('speed',speeds,2); fill('trail',trails,0); fill('step',steps,1);
const val = id => +document.getElementById(id).value;

// great-circle arc between two [lon,lat] points, lifted into space and back
function arc(p,q,n=40){
  const r=Math.PI/180, [l1,f1]=[p[0]*r,p[1]*r], [l2,f2]=[q[0]*r,q[1]*r];
  const d=2*Math.asin(Math.sqrt(Math.sin((f2-f1)/2)**2+Math.cos(f1)*Math.cos(f2)*Math.sin((l2-l1)/2)**2));
  if(d<1e-6) return [[p[0],p[1],0],[q[0],q[1],0]];
  const H=d*6371000*0.22; // apex height in metres
  const out=[];
  for(let i=0;i<=n;i++){const k=i/n,A=Math.sin((1-k)*d)/Math.sin(d),B=Math.sin(k*d)/Math.sin(d);
    const x=A*Math.cos(f1)*Math.cos(l1)+B*Math.cos(f2)*Math.cos(l2);
    const y=A*Math.cos(f1)*Math.sin(l1)+B*Math.cos(f2)*Math.sin(l2);
    const z=A*Math.sin(f1)+B*Math.sin(f2);
    out.push([Math.atan2(y,x)/r,Math.atan2(z,Math.hypot(x,y))/r,Math.sin(Math.PI*k)*H]);}
  return out;
}
function at(line,f){f=Math.max(0,Math.min(1,f||0));const x=f*(line.length-1),i=Math.min(Math.floor(x),line.length-2),u=x-i;
  return [0,1,2].map(c=>line[i][c]+(line[i+1][c]-line[i][c])*u)}
const fmt = t => new Date(t*1000).toISOString().slice(0,10);

class Panel {
  constructor(D, host){
    this.D = D; this.arcCache = new Map();
    host.innerHTML = `<div class="map"></div><div class="hud"><div class="name"></div><div class="date">…</div><div class="stats"></div><div class="top"></div></div>`;
    this.el = host; this.q = c => host.querySelector('.'+c);
    this.q('name').textContent = D.name;
    this.map = new maplibregl.Map({
      container:this.q('map'), center:[10,30], zoom:0.6, attributionControl:{compact:true},
      style:{version:8,
        sources:{sat:{type:'raster',tileSize:256,maxzoom:18,attribution:'Imagery © Esri, Maxar, Earthstar Geographics',
          tiles:['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}']}},
        layers:[{id:'sat',type:'raster',source:'sat'}]}
    });
    this.map.on('style.load',()=>{ try{this.map.setProjection({type:'globe'})}catch(e){console.warn(e)} });
    this.overlay = new deck.MapboxOverlay({interleaved:true,layers:[]});
    this.map.addControl(this.overlay);
    // camera in tandem: a user move on one globe is mirrored onto the others
    this.map.on('move',e=>{
      if(syncing||!e.originalEvent) return;
      syncing=true;
      const c={center:this.map.getCenter(),zoom:this.map.getZoom(),bearing:this.map.getBearing(),pitch:this.map.getPitch()};
      panels.forEach(p=>{if(p!==this)p.map.jumpTo(c)});
      syncing=false;
    });
  }
  lb(x){const t=this.D.t;let lo=0,hi=t.length;while(lo<hi){const m=(lo+hi)>>1;if(t[m]<x)lo=m+1;else hi=m}return lo}
  route(a,b){const k=a*1000+b;let v=this.arcCache.get(k);
    if(!v){const A=this.D.countries[a],B=this.D.countries[b];v=arc([A.lon,A.lat],[B.lon,B.lat]);this.arcCache.set(k,v)}return v}
  render(){
    const D=this.D, trail=Math.max(1,val('trail')*val('speed'))*DAY, flight=Math.max(2,val('speed')*1.5)*DAY;
    const i0=this.lb(now-flight-trail), i1=this.lb(now+1);
    const agg=new Map();
    for(let i=i0;i<i1;i++){const k=D.a[i]*1000+D.b[i];let g=agg.get(k);
      if(!g){g={a:D.a[i],b:D.b[i],n:0,t:0};agg.set(k,g)} g.n++; g.t=D.t[i];}
    const lines=[], top=[];
    for(const g of agg.values()){
      lines.push({path:this.route(g.a,g.b),n:g.n,age:Math.max(0,(now-g.t-flight)/trail)});
      top.push(g);
    }
    const dots=[], j0=this.lb(now-flight), stride=Math.max(1,Math.ceil((i1-j0)/1500));
    for(let i=j0;i<i1;i+=stride){const f=(now-D.t[i])/flight;
      dots.push({pos:at(this.route(D.a[i],D.b[i]),Math.min(1,f)),f});}
    this.overlay.setProps({layers:[
      ...(ARCS?[new deck.PathLayer({id:'arcs',data:lines,getPath:d=>d.path,
        getColor:d=>{const k=1-Math.min(1,d.age);return [255*k,215*k,0,Math.round(0.2*255*k)]},
        getWidth:d=>2.5+1.5*Math.log2(1+d.n),widthUnits:'pixels',widthMinPixels:2.5,capRounded:true,
        getDashArray:[5,3],dashJustified:false,extensions:[new deck.PathStyleExtension({dash:true})],
        updateTriggers:{getColor:[now]}})]:[]),
      new deck.ScatterplotLayer({id:'dots',data:dots,getPosition:d=>d.pos,
        getFillColor:d=>[255,250,210,Math.round(255-170*d.f)],radiusUnits:'pixels',getRadius:6,
        parameters:{depthCompare:'always'},updateTriggers:{getFillColor:[now]}})
    ]});
    top.sort((x,y)=>y.n-x.n);
    this.q('date').textContent=fmt(now);
    this.q('stats').innerHTML=`${i1-i0} transfers shown<br>${agg.size} active routes`;
    this.q('top').innerHTML=top.slice(0,5).map(g=>`${D.countries[g.a].code} → ${D.countries[g.b].code}: ${g.n}`).join('<br>');
  }
}

function frame(){
  panels.forEach(p=>p.render());
  document.getElementById('slider').value=Math.round((now-T0)/(T1-T0)*10000);
}
function setTime(t){now=Math.max(T0,Math.min(T1,t));frame()}
function setPlaying(p){playing=p;last=0;document.getElementById('bPlay').textContent=p?'⏸':'▶';
  document.getElementById('bPlay').classList.toggle('on',p)}
function tick(ts){
  if(playing){if(last){const dt=(ts-last)/1000;setTime(now+dt*val('speed')*DAY);
    if(now>=T1) setPlaying(false);}
    last=ts;}
  requestAnimationFrame(tick);
}
const stepBy = (dir,big) => {setTime(now+dir*val('step')*DAY*(big?4:1))};
const stop = () => {setPlaying(false);setTime(T0)};
const toggle = () => {if(!playing&&now>=T1)setTime(T0);setPlaying(!playing)};
document.getElementById('bBack').onclick=()=>stepBy(-1);
document.getElementById('bFwd').onclick=()=>stepBy(1);
document.getElementById('bStop').onclick=stop;
document.getElementById('bPlay').onclick=toggle;
document.getElementById('slider').oninput=e=>setTime(T0+e.target.value/10000*(T1-T0));
document.getElementById('trail').onchange=frame;
document.getElementById('speed').onchange=frame;
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'&&e.target.type!=='range'||e.metaKey||e.ctrlKey)return;
  const k=e.key; let h=true;
  if(k===' ')toggle(); else if(k==='s'||k==='S')stop();
  else if(k==='ArrowLeft')stepBy(-1,e.shiftKey); else if(k==='ArrowRight')stepBy(1,e.shiftKey);
  else if(k>='1'&&k<='9'){document.getElementById('speed').selectedIndex=+k-1;frame()}
  else if(k==='Home')setTime(T0); else if(k==='End')setTime(T1); else h=false;
  if(h){e.preventDefault();if(document.activeElement&&document.activeElement.tagName!=='BODY')document.activeElement.blur()}
});

(async()=>{
  const resp = await (await fetch('/api/transfers')).json();
  const sets = resp.datasets; ARCS = resp.options.arcs;
  const host = document.getElementById('panels');
  const divs = sets.map(()=>{const d=document.createElement('div');d.className='panel';host.appendChild(d);return d});
  panels = sets.map((D,i)=>new Panel(D,divs[i]));  // all panels laid out before any map measures its container
  T0 = Math.min(...sets.map(D=>D.t[0])); T1 = Math.max(...sets.map(D=>D.t[D.t.length-1])); now = T0;
  await Promise.all(panels.map(p=>new Promise(r=>p.map.loaded()?r():p.map.once('load',r))));
  frame(); requestAnimationFrame(tick);
})();
</script></body></html>"""


def create_app(datasets, options):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/transfers")
    def transfers():
        return jsonify({"datasets": datasets, "options": options})

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--left", "-i", dest="left", required=True, help="country_transfer.tsv (left globe)")
    ap.add_argument("--right", "--other", dest="right", help="optional second TSV, shown on a right-hand globe in tandem (e.g. synth/travel_synth.tsv)")
    ap.add_argument("--disable-arc", action="store_true", help="hide the arcs; show only the moving dots")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5055)
    args = ap.parse_args()
    first = load(args.left)
    first["name"] = os.path.basename(args.left)
    datasets = [first]
    if args.right:
        second = load(args.right, {c["code"]: [c["lat"], c["lon"]] for c in first["countries"]})
        second["name"] = os.path.basename(args.right)
        datasets.append(second)
    for d in datasets:
        print(f"{d['name']}: {len(d['t'])} transfers, {len(d['countries'])} countries")
    print(f"-> http://{args.host}:{args.port}")
    create_app(datasets, {"arcs": not args.disable_arc}).run(host=args.host, port=args.port, debug=False, use_reloader=True)
