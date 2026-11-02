#!/bin/bash
# Throwaway: lever sweep for attr_degree (share / bins). Scratch dirs only; shipped outputs untouched.
set -e
cd "$(dirname "$0")/../.."
for name in share35 bins4 share35_bins4; do
  echo "== $name: stage 1 =="
  PYTHONPATH=src:snap/inside python3 snap/inside/fit_artifacts.py --seeds 3 \
    --config tools/sweep/$name.json --out tools/sweep/$name/artifacts \
    --only graph_aim_eps4 graph_aim_epsinf
  echo "== $name: stage 2 =="
  PYTHONPATH=src python3 snap/outside/generate_synthetic.py \
    --artifacts tools/sweep/$name/artifacts --out tools/sweep/$name/synthetic
  echo "== $name: coupling =="
  PYTHONPATH=src python3 snap/eval/coupling_report.py \
    --real snap/inside/data/processed/attributed_graph.npz \
    --syn tools/sweep/$name/synthetic/graph_aim_eps4_seed0.npz \
          tools/sweep/$name/synthetic/graph_aim_eps4_seed1.npz \
          tools/sweep/$name/synthetic/graph_aim_eps4_seed2.npz \
          tools/sweep/$name/synthetic/graph_aim_epsinf_seed0.npz \
          tools/sweep/$name/synthetic/graph_aim_epsinf_seed1.npz \
          tools/sweep/$name/synthetic/graph_aim_epsinf_seed2.npz \
    --bin-edges 0 1 4 16 65 260 1046 --exclude-hubs 10 \
    --out tools/sweep/$name/coupling.csv > tools/sweep/$name/coupling_report.log
  echo "== $name done =="
done
