"""Stable degree cap: moved to ``graph_aim.transfer.capping`` (composition round, v2).

The generator now caps the synthetic graph exactly as the released statistics were capped, so the cap lives in
the transfer zone where both sides can import it. This module re-exports it so inside code and tests are unchanged.
"""

from __future__ import annotations

from graph_aim.transfer.capping import *  # noqa: F401,F403
from graph_aim.transfer.capping import cap_edges, edge_hash

__all__ = ["cap_edges", "edge_hash"]
