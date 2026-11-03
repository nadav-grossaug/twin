#!/bin/bash
# Throwaway: package-D decision-point sweep. Scratch dirs only; shipped outputs untouched.
set -e
cd "$(dirname "$0")/../.."
for name in X_single Y_three Z_two; do
  echo "== $name: stage 1 =="
  PYTHONPATH=src:snap/inside python3 snap/inside/fit_artifacts.py --seeds 3 \
    --config tools/sweep_d/$name.json --out tools/sweep_d/$name/artifacts \
    --only graph_aim_eps4 graph_aim_epsinf
  echo "== $name: stage 2 =="
  PYTHONPATH=src python3 snap/outside/generate_synthetic.py \
    --artifacts tools/sweep_d/$name/artifacts --out tools/sweep_d/$name/synthetic
  echo "== $name: coupling =="
  PYTHONPATH=src python3 snap/eval/coupling_report.py \
    --real snap/inside/data/processed/attributed_graph.npz \
    --syn tools/sweep_d/$name/synthetic/graph_aim_eps4_seed0.npz \
          tools/sweep_d/$name/synthetic/graph_aim_eps4_seed1.npz \
          tools/sweep_d/$name/synthetic/graph_aim_eps4_seed2.npz \
          tools/sweep_d/$name/synthetic/graph_aim_epsinf_seed0.npz \
          tools/sweep_d/$name/synthetic/graph_aim_epsinf_seed1.npz \
          tools/sweep_d/$name/synthetic/graph_aim_epsinf_seed2.npz \
    --bin-edges 0 1 4 16 65 260 1046 --exclude-hubs 10 \
    --out tools/sweep_d/$name/coupling.csv > tools/sweep_d/$name/coupling_report.log
  echo "== $name: score =="
  PYTHONPATH=src:snap/eval python3 snap/eval/score.py --seeds 3 \
    --synthetic tools/sweep_d/$name/synthetic --out tools/sweep_d/$name/results \
    --only graph_aim > tools/sweep_d/$name/score.log 2>&1
  echo "== $name done =="
done
