"""CLI: fit, sample and evaluate Synthcity generators on a registered dataset.

    python -m synth_bench.run_benchmark --dataset adult-census --methods all --n-train 10000

Outputs per method under ``<results-dir>/<method>/``: ``metrics.json`` and
``synthetic.csv``; ``<results-dir>/summary.csv`` has one row per method found on disk
(so partial and resumed runs accumulate). A method that raises is recorded with
``status = "error"`` and the run continues.
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import pandas as pd

from synth_bench import config
from synth_bench.datasets import get_dataset, list_datasets

log = logging.getLogger("synth_bench")


def parse_methods(arg: str, available: list[str]) -> list[str]:
    """Parse ``"all"`` or a comma-separated list, validating the names."""
    if arg.strip() == "all":
        return list(available)
    names = [m.strip() for m in arg.split(",") if m.strip()]
    unknown = [m for m in names if m not in available]
    if unknown:
        raise SystemExit(f"Unknown method(s) {unknown}; available: {available}")
    return names


def _flatten(metrics: dict) -> dict:
    """Keep scalar metrics only (per-column dicts stay in metrics.json)."""
    return {k: v for k, v in metrics.items() if not isinstance(v, dict | list)}


def run_method(method, spec, train, test, out_dir: Path, n_synth: int, seed: int,
               synthcity_metrics: bool = True) -> dict:
    """Fit/generate/evaluate one method; always writes ``metrics.json``. Returns it."""
    from synth_bench.data import to_loader
    from synth_bench.evaluate import evaluate
    from synth_bench.generators import fit_generate
    from synth_bench.methods import get_params

    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"method": method, "dataset": spec.name, "seed": seed, "n_train": len(train)}
    try:
        params = get_params(method, spec)
        record["params"] = params
        synth, fit_s, gen_s = fit_generate(
            method, to_loader(train, spec), params, n_synth, seed, spec.discrete_columns
        )
        record.update(fit_seconds=fit_s, generate_seconds=gen_s)
        synth.to_csv(out_dir / "synthetic.csv", index=False)
        record.update(evaluate(train, test, synth, spec, seed, synthcity_metrics))
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a failing method must not abort the run
        log.error("method %s failed: %s", method, exc)
        record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def write_summary(results_dir: Path) -> pd.DataFrame:
    """Rebuild ``summary.csv`` from every ``<method>/metrics.json`` under ``results_dir``."""
    rows = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        rec = json.loads(path.read_text())
        row = _flatten({k: v for k, v in rec.items() if k != "traceback"})
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        first = [c for c in ("method", "status") if c in summary.columns]
        summary = summary[first + [c for c in summary.columns if c not in first]]
    summary.to_csv(results_dir / "summary.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    from synth_bench.data import load_splits
    from synth_bench.methods import list_methods

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default="adult-census", choices=list_datasets())
    p.add_argument("--methods", default="all", help="comma-separated names or 'all'")
    p.add_argument("--n-train", type=int, default=config.N_TRAIN,
                   help="rows to fit on (0 = the full train split)")
    p.add_argument("--n-synth", type=int, default=config.N_SYNTH,
                   help="rows to generate (default: same as n-train)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--results-dir", type=Path, default=None,
                   help="output folder (default: <dataset>/results)")
    p.add_argument("--resume", action="store_true",
                   help="skip methods that already have an ok metrics.json")
    p.add_argument("--no-synthcity-metrics", action="store_true",
                   help="skip k-anonymity / l-diversity from Synthcity")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    spec = get_dataset(args.dataset)
    methods = parse_methods(args.methods, list_methods())
    results_dir = args.results_dir or config.results_dir(args.dataset)
    results_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_splits(args.dataset, args.n_train or None, args.seed)
    n_synth = args.n_synth or len(train)
    log.info("%s: train=%d test=%d n_synth=%d methods=%s", spec.name, len(train), len(test),
             n_synth, methods)

    failed = []
    for method in methods:
        out_dir = results_dir / method
        done = out_dir / "metrics.json"
        if args.resume and done.exists() and json.loads(done.read_text()).get("status") == "ok":
            log.info("[%s] skipped (resume)", method)
            continue
        log.info("[%s] running", method)
        rec = run_method(method, spec, train, test, out_dir, n_synth, args.seed,
                         not args.no_synthcity_metrics)
        if rec["status"] != "ok":
            failed.append(method)
        log.info("[%s] %s", method, rec["status"])
        write_summary(results_dir)  # keep summary.csv current if the run is interrupted

    summary = write_summary(results_dir)
    log.info("summary written to %s (%d rows); failed: %s", results_dir / "summary.csv",
             len(summary), failed or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
