"""Human-readable report on a graph-AIM artifact: ``python -m graph_aim.transfer.inspect <artifact_dir>``.

Note: this module is named ``inspect`` like the stdlib module. It only shadows the stdlib if the
``graph_aim`` package directory itself is placed on ``sys.path``; with ``src`` on the path (or an
installed package) the stdlib is untouched. All imports here are absolute for that reason.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from graph_aim.transfer.artifact import load, validate, zcdp_to_eps
from graph_aim.transfer.types import Artifact


def _fmt(x: float, digits: int = 4) -> str:
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}g}"


def noise_to_signal(m) -> float:
    """sigma divided by the mean cell magnitude of the released values (inf if the mean is 0)."""
    mean = float(np.mean(np.abs(m.values))) if np.size(m.values) else 0.0
    if m.sigma == 0:
        return 0.0
    return math.inf if mean == 0 else m.sigma / mean


def inspect_report(path_or_artifact) -> str:
    """Markdown report. A path is loaded with full hash verification (raises if tampered)."""
    art = path_or_artifact if isinstance(path_or_artifact, Artifact) else load(path_or_artifact)
    s, led = art.schema, art.ledger
    out = ["# Graph-AIM artifact report", ""]

    out += ["## Schema", "", f"- nodes (public): {s.n}", f"- degree cap D: {s.degree_cap}", f"- cap salt: {s.cap_salt}"]
    out += [f"- homophily attributes: {', '.join(s.homophily_attrs) or 'none'}", "", "| attribute | domain size | labels |", "|---|---|---|"]
    for col, labels in s.domains.items():
        out.append(f"| {col} | {len(labels)} | {', '.join(labels)} |")

    out += ["", "## Privacy budget", "", f"delta = {_fmt(led.delta)}. Conversion: eps = rho + 2 sqrt(rho ln(1/delta)).", ""]
    out += ["| adjacency | rho (zCDP) | epsilon |", "|---|---|---|"]
    out.append(f"| edge | {_fmt(led.rho_edge)} | {_fmt(zcdp_to_eps(led.rho_edge, led.delta))} |")
    out.append(f"| attribute row | {_fmt(led.rho_attr)} | {_fmt(zcdp_to_eps(led.rho_attr, led.delta))} |")
    if math.isinf(led.rho_edge) or math.isinf(led.rho_attr):
        out += ["", "WARNING: infinite rho, this artifact is NOT differentially private (non-private upper bound)."]
    out += ["", "The guarantee holds separately for each adjacency; nothing is claimed for simultaneous changes.", ""]

    out += [
        "## Measurements",
        "",
        "Noise-to-signal = sigma / mean absolute cell value (lower is better).",
        "",
        "| name | kind | shape | sigma | sens. edge | sens. attr | rho edge | rho attr | noise/signal |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_name = {e.get("name"): e for e in led.entries}
    for name, m in art.measurements.items():
        e = by_name.get(name, {})
        out.append(
            f"| {name} | {m.kind} | {'x'.join(map(str, np.shape(m.values)))} | {_fmt(m.sigma)} | "
            f"{_fmt(m.sensitivity_edge)} | {_fmt(m.sensitivity_attr)} | {_fmt(e.get('rho_edge', float('nan')))} | "
            f"{_fmt(e.get('rho_attr', float('nan')))} | {_fmt(noise_to_signal(m))} |"
        )

    out += ["", "## Caveats (not covered by the guarantee)", ""]
    out += [f"- {c}" for c in led.caveats] or ["- none recorded"]

    problems = validate(art)
    out += ["", "## Validation", ""]
    out += [f"- {p}" for p in problems] or ["- no problems found"]
    if art.provenance:
        out += ["", "## Provenance", ""] + [f"- {k}: {v}" for k, v in art.provenance.items()]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m graph_aim.transfer.inspect <artifact_dir>", file=sys.stderr)
        return 2
    try:
        print(inspect_report(Path(argv[0])))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
