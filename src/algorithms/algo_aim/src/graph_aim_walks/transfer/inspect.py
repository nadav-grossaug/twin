"""Human-readable report on a graph-aim-walks artifact: ``python -m graph_aim_walks.transfer.inspect <artifact_dir>``.

Named ``inspect`` like the stdlib module (and like ``graph_aim.transfer.inspect``); it only shadows the stdlib if the
package directory itself is on ``sys.path``. All imports are absolute for that reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from graph_aim_walks.transfer.artifact import PRIVACY_WARNING, load, validate
from graph_aim_walks.transfer.types import SPECTRUM, WalksArtifact


def _fmt(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}g}"


def _at(v: np.ndarray, steps) -> str:
    return ", ".join(f"t={t}: {_fmt(v[t - 1])}" for t in steps if 1 <= t <= len(v))


def inspect_report(path_or_artifact) -> str:
    """Markdown report. A path is loaded with full hash verification (raises if tampered)."""
    art = path_or_artifact if isinstance(path_or_artifact, WalksArtifact) else load(path_or_artifact)
    s, spec, tb = art.base.schema, art.spec, art.tables
    out = ["# graph-aim-walks artifact report", "", f"**{PRIVACY_WARNING}**", ""]

    out += ["## Walk spec", "", f"- walks: {art.n_walks}, length T = {spec.length}", f"- lags: {list(spec.lags)}",
            f"- windows: {list(spec.windows)}, share bins: {spec.share_bins}, dominance bins: {spec.dominance_bins}",
            f"- degree bin edges: {list(spec.degree_bin_edges)}",
            f"- attr_lag groups: {', '.join('+'.join(g) for g in spec.groups) or 'none'}",
            f"- share attributes: {', '.join(spec.share_attrs) or 'none'}",
            f"- dominance pairs: {', '.join(f'{a}|{b}' for a, b in spec.dominance_pairs) or 'none'}", ""]

    out += ["## Tables", "", "| name | shape | min | max |", "|---|---|---|---|"]
    for name, v in tb.items():
        v = np.asarray(v)
        out.append(f"| {name} | {'x'.join(map(str, v.shape))} | {_fmt(v.min()) if v.size else '-'} | "
                   f"{_fmt(v.max()) if v.size else '-'} |")

    T = spec.length
    marks = [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= T]
    out += ["", "## Highlights", ""]
    if "return" in tb:
        out.append(f"- return probability: {_at(np.asarray(tb['return']), marks)}")
    if "coverage" in tb:
        out.append(f"- distinct nodes visited (mean): {_at(np.asarray(tb['coverage']), marks)}")
    for name, v in tb.items():
        if name.startswith("attr_lag:"):
            same = np.trace(np.asarray(v), axis1=1, axis2=2)
            out.append(f"- {name}: same group at lag " + ", ".join(f"{k}: {_fmt(x, 3)}" for k, x in zip(spec.lags, same)))
    if SPECTRUM in tb:
        sv = np.asarray(tb[SPECTRUM])
        out.append(f"- transition spectrum (top {min(10, len(sv))} of {len(sv)}): "
                   + ", ".join(_fmt(x) for x in sv[:10]))

    out += ["", "## Base graph-AIM artifact (exact, noise off)", "", f"- nodes: {s.n}",
            f"- attributes: {', '.join(f'{c} ({len(d)})' for c, d in s.domains.items())}",
            f"- measurements: {', '.join(art.base.measurements)}"]

    problems = validate(art)
    out += ["", "## Validation", ""]
    out += [f"- {p}" for p in problems] or ["- no problems found"]
    if art.provenance:
        out += ["", "## Provenance", ""] + [f"- {k}: {v}" for k, v in art.provenance.items()]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m graph_aim_walks.transfer.inspect <artifact_dir>", file=sys.stderr)
        return 2
    try:
        print(inspect_report(Path(argv[0])))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
