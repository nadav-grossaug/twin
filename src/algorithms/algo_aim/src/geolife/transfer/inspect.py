"""Human-readable artifact report (plots live in eval/plots.py: they need matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geolife.transfer.artifact import Artifact


def _fmt_table(rows: list[list[str]], md: bool) -> list[str]:
    if not rows:
        return []
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for n, r in enumerate(rows):
        if md:
            out.append("| " + " | ".join(r) + " |")
            if n == 0:
                out.append("|" + "|".join("---" for _ in r) + "|")
        else:
            out.append("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
    return out


def _vocab_sizes(art: Artifact) -> dict[str, int]:
    """Distinct values per key position (hierarchy level), per measurement."""
    out = {}
    for m in art.measurements:
        width = max((len(k) for k in m.keys), default=0)
        for i in range(width):
            out[f"{m.name}[{i}]"] = len({k[i] for k in m.keys if len(k) > i})
    return out


def render_report(art: Artifact, markdown: bool = False, top_k: int = 5) -> str:
    h = (lambda s: f"## {s}") if markdown else (lambda s: f"== {s} ==")
    L: list[str] = []
    mf = art.manifest
    L += [h("Manifest"), f"format {mf.format_version}, learner {mf.learner}, "
          f"map {mf.source_map_snapshot or '?'}, H3 res {mf.h3_resolution}, "
          f"created {mf.created or '?'}, {len(mf.files)} files", ""]
    L += [h("Schema")]
    for k, v in art.schema.items():
        s = f"{len(v)} entries" if isinstance(v, (dict, list)) else repr(v)
        if isinstance(v, dict) and len(v) <= 8:
            s += ": " + ", ".join(map(str, v))
        L.append(f"- {k}: {s}")
    L += ["", h("Vocabulary sizes per hierarchy level")]
    vs = _vocab_sizes(art)
    L += [f"- {k}: {n}" for k, n in vs.items()] or ["(none)"]
    L += ["", h("Measurements (top keys, noisy value vs sigma)")]
    for m in art.measurements:
        L.append(f"{m.name}: {len(m.keys)} keys, sigma={m.sigma:g}, "
                 f"L2 sens={m.l2_sensitivity:g}, rho={m.rho:g}" + (f"  [{m.notes}]" if m.notes else ""))
        order = sorted(range(len(m.keys)), key=lambda i: -m.values[i])[:top_k]
        rows = [["key", "value", "sigma", "flag"]]
        for i in order:
            v = m.values[i]
            rows.append(["/".join(m.keys[i]), f"{v:.2f}", f"{m.sigma:.2f}",
                         "LOW-SNR" if v < 2 * m.sigma else "ok"])
        L += _fmt_table(rows, markdown)
        low = sum(v < 2 * m.sigma for v in m.values)
        L += [f"  {low}/{len(m.values)} keys below 2*sigma", ""]
    led = art.ledger
    L += [h("Privacy ledger"), f"adjacency={led.adjacency}  rho_total={led.rho_total:g}  "
          f"epsilon={led.epsilon:g}  delta={led.delta:g}"]
    if led.caps:
        L.append("caps: " + ", ".join(f"{k}={v}" for k, v in led.caps.items()))
    rows = [["name", "rho", "sigma", "sensitivity"]]
    for e in led.entries:
        rows.append([str(e.get("name")), f"{e.get('rho', 0):g}", f"{e.get('sigma', 0):g}",
                     f"{e.get('l2_sensitivity', 0):g}"])
    L += _fmt_table(rows, markdown)
    L += ["Caveats:"] + [f"- {c}" for c in led.caveats] if led.caveats else ["Caveats: (none stated)"]
    L += ["", h("Embeddings")]
    L += [f"- {k}: shape {tuple(v.shape)} {v.dtype}" for k, v in art.embeddings.items()] or ["(none)"]
    return "\n".join(L) + "\n"
