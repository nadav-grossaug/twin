"""Throwaway (package E): which attr_edge_pair instance is primary on each Y_three artifact?"""

from graph_aim.outside.generate import _by_kind, _edge_pair_lift, _primary_edge_pair
from graph_aim.transfer.artifact import load

for eps in ["eps4", "epsinf"]:
    for seed in range(3):
        variant = f"graph_aim_{eps}_seed{seed}"
        art = load(f"tools/sweep_d/Y_three/artifacts/{variant}")
        lifts = {m.name: round(_edge_pair_lift(m.values), 3) for m in _by_kind(art, "attr_edge_pair")}
        print(variant, lifts, "-> primary:", _primary_edge_pair(art).name)
