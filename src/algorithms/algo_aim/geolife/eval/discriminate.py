"""STAGE 3 (eval): real + synthetic traces -> real-vs-fake discriminator. Thin entry point; the code
is src/geolife/eval/discriminate.py.

    PYTHONPATH=src python geolife/eval/discriminate.py [--n 4000] [--models logreg gbm] [--no-map]
        [--points geolife/inside/data/points.parquet] [--synthetic geolife/outside/synthetic]
        [--places geolife/outside/cache/places_beijing_big.parquet] [--out geolife/eval/results/discriminator]
        [--folds 3] [--max-users 20]    # --max-users for a quick smoke run

Eval is an inside-only zone: it reads real traces. Never copy its results across the boundary without review.
"""
import sys

from geolife.eval.discriminate import main

if __name__ == "__main__":
    sys.exit(main())
