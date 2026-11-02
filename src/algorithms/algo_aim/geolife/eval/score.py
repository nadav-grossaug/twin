"""STAGE 3 (eval): synthetic + real -> metrics. Thin entry point; the code is src/geolife/eval/score.py.

    PYTHONPATH=src python geolife/eval/score.py [--synthetic geolife/outside/synthetic] [--out geolife/eval/results]
        [--real-points geolife/inside/data/points.parquet --places geolife/outside/cache/places_beijing_big.parquet]
"""
import sys

from geolife.eval.score import main

if __name__ == "__main__":
    sys.exit(main())
