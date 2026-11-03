"""Throwaway (package E): are two synthetic-graph folders identical graph by graph?"""

import sys

import numpy as np

from graph_aim.transfer.graph_io import load_attributed_graph

a_dir, b_dir = sys.argv[1], sys.argv[2]
for eps in ["eps4", "epsinf"]:
    for seed in range(3):
        name = f"graph_aim_{eps}_seed{seed}.npz"
        a = load_attributed_graph(f"{a_dir}/{name}")
        b = load_attributed_graph(f"{b_dir}/{name}")
        same_e = np.array_equal(a.edges, b.edges)
        same_a = a.attrs.equals(b.attrs)
        print(name, "edges_equal:", same_e, "attrs_equal:", same_a, f"m_a={len(a.edges)} m_b={len(b.edges)}")
