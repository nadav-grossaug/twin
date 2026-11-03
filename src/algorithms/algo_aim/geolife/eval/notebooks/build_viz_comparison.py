"""Builds geolife/eval/notebooks/viz_comparison.ipynb (python build_viz_comparison.py; execute with
geolife/inside/notebooks/run_nb.py from the repo root, cwd = this folder).

The notebook is thin: it builds the two `TraceBundle`s and calls `geolife.eval.viz` /
`geolife.eval.compare` (modelled on snap/eval/notebooks/viz_comparison.ipynb). Inputs are the real
points parquet, the public places parquet and one stage-2 output directory; override with the env
vars GEOLIFE_POINTS, GEOLIFE_PLACES, GEOLIFE_SYN.
"""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Viz comparison: real GeoLife traces against synthetic traces

How the real Beijing traces and the synthetic traces generated outside from a DP artifact behave, side by
side. **Black is always the real data**; the coloured line is the synthetic variant. This notebook lives in
`geolife/eval/` because it needs both: the real points (`geolife/inside/data/points.parquet`) and a stage-2
output (`geolife/outside/synthetic/<variant>_seed<k>/`). The library code is `src/geolife/eval/viz.py`
(panels) and `src/geolife/eval/compare.py` (quantities, moments, distances) - a new statistic is one line
in `compare.QUANTITIES`.

For the **direct lon/lat overlay** the synthetic variant must be generated into the source city:
`PYTHONPATH=src python geolife/outside/generate_synthetic.py --only outings_v2_seed0 --city beijing --n-users 139`.
A teleported variant (other city) is drawn in centred metric frames instead (same scale, km from each
dataset's centroid); the shape panels do not depend on the frame at all.

Both sides go through the **same tokenizer** (`transfer/tokenize/outings.py`, public Beijing places table),
so outing-level gaps are generator gaps, not tokenizer artefacts. Times are local (UTC+8).""")

code("""import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings("ignore")
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "geolife").exists())
sys.path.insert(0, str(ROOT / "src"))
from geolife.eval import compare as C
from geolife.eval import viz as V
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50)

POINTS = os.environ.get("GEOLIFE_POINTS", str(ROOT / "geolife/inside/data/points.parquet"))
PLACES = os.environ.get("GEOLIFE_PLACES", str(ROOT / "geolife/outside/cache/places_beijing_big.parquet"))
SYN    = os.environ.get("GEOLIFE_SYN",    str(ROOT / "geolife/outside/synthetic/outings_v2_seed0"))
print(POINTS, "\\n" + PLACES, "\\n" + SYN)""")

code("""places = pd.read_parquet(PLACES)
real = C.bundle_from_points_parquet(POINTS, label="real", places=places)       # Beijing core bbox
syn = C.bundle_from_synthetic(SYN, label=Path(SYN).name, places=places)
syns = {syn.label: syn}
for b in (real, syn):
    print(f"{b.label}: {len(b.arrays)} users, {sum(len(a[0]) for a in b.arrays.values()):,} points, "
          f"{len(b.outings)} tokenized users, {sum(len(u.outings) for u in b.outings):,} outings")""")

md("""## 1. Space

Density of all GPS points, the difference of the two normalised densities (orange = synthetic excess,
blue = real excess), the marginal lon/lat histograms with KS, and a few example users' raw traces at the
same scale. *Look for:* does the synthetic mass sit on the same corridors and hubs, and do single users'
traces have the same visual texture (noise, stops, straight transit runs)?""")
code("V.density_maps(real, syns)")
code("V.marginal_histograms(real, syns)")
code("V.example_trajectories(real, syns, n=5)")

md("""**What we see (baseline, before sensor tables).** The synthetic mass sits on the real road network
(teleport works: same ring roads, same hub quarters), but it concentrates south-east of the real centre of
mass and spreads wider; the marginals overlap with small KS. The example traces give it away: real traces
are thin lines with sharp corners, synthetic ones are fuzzy bands - the hand-set GPS noise is large
relative to the 2 s step.""")

md("""## 2. Moments and shape

ECDF overlays and QQ plots for every numeric quantity: point level (sampling interval, gap length, step
length, speed overall and by mode, heading change, positional jitter at stops), outing level (length,
duration, stops, dwell) and user level (points, outings, active days, distinct places, revisit rate).
*Look for:* bulk agreement in the ECDFs; QQ points on the diagonal (above = synthetic heavier tail).""")
code("V.ecdf_panels(real, syns)")
code("V.qq_panels(real, syns)")

md("""**What we see.** The biggest gaps are exactly the hand-set sensor knobs (PLAN next step 1): heading
change (real median ~4 deg, synthetic ~120 deg: noise dominates the direction signal), speeds (synthetic
"walk" at ~14 m/s because rendered speeds are too high and the mode heuristic reclassifies), step length
~2x, gaps too long and too rare, stop jitter too clean. Outing length/duration and points per user are
several times too large (generator realism, step 5). Sampling interval, stops per outing and outings per
user are already close.""")

md("""## 3. Time

Hour-of-day profiles of outing starts and stops (weekday / weekend, local time) and the coarse time-bin
shares. *Look for:* the real 08h and 17-18h start peaks and the weekend skew of stops.""")
code("V.hour_profiles(real, syns)")
code("V.timebin_shares(real, syns)")

md("""**What we see.** The synthetic day is too flat: the sharp real 08h start peak and the 17-18h shoulder
are smeared (hour-profile TV 0.22-0.33, time-bin shares TV 0.18) - expected while the released time tables
are mostly noise at epsilon 6 (3-bin redesign is PLAN next step 2).""")

md("""## 4. Social / place structure

Users per place and per traversed cell (CCDF), revisit rate per user, and the co-visit degree
distribution; then the same panels on split halves of each dataset (A solid, B dashed) - a dataset whose
halves disagree is unstable before it is wrong. *Look for:* hub places reaching tens of users in the real
curve, and the synthetic co-visit degrees not exceeding the real ones.""")
code("V.structure_panels(real, syns)")
code("V.split_half_structure(real, syns)")

md("""**What we see.** Structure is the closest part of the baseline: users per place TV 0.08 and users per
cell TV 0.09 with the same power-law shape and hub tail, co-visit degree a little high (mean 24 vs 19).
Revisit rate is clearly shifted (0.57 vs 0.45, and no low-revisit users at all). The split halves agree
within each dataset, so these gaps are real, not noise.""")

md("""## 5. Summary distance table

One row per quantity: moments of both sides (mean, std, skewness, kurtosis, p1..p99 in the CSV) and the
distances (KS, Wasserstein-1, total variation, relative error of moments). Written to
`geolife/eval/results/compare_<variant>.{csv,md}` so progress can be tracked run to run.""")
code("""df = C.summary_table(real, syn)
C.write_summary(df, ROOT / "geolife" / "eval" / "results", f"compare_{Path(SYN).name}")
df[[c for c in C.MD_COLS if c in df.columns]].round(4)""")

nb = nbf.v4.new_notebook(cells=C, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                           "language": "python"}})
nbf.write(nb, "viz_comparison.ipynb")
print("wrote viz_comparison.ipynb")
