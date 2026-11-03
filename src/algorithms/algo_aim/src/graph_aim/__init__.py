"""Graph-AIM: inspectable, differentially private synthetic attributed graphs.

Three zones, see snap/README.md:
  inside    code that may read the real graph (fit, measure, privacy, evaluation, baselines, attacks)
  transfer  what crosses the boundary (shared types, artifact format, graph file format)
  outside   generation from an artifact; never sees the real graph
"""
