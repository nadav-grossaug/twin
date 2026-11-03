"""Shipped map code never fetches: a cache miss raises unless the caller injects a `fetch` callable."""
import pandas as pd
import pytest

from geolife.outside.map.overture import load_overture
from geolife.outside.map.provider import fetch_places

BBOX = (4.85, 52.33, 4.95, 52.41)


def test_cache_miss_without_fetcher_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_public_map"):
        fetch_places(BBOX, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_public_map"):
        load_overture("segment", BBOX, cache_dir=tmp_path)


def test_injected_fetcher_fills_the_cache_once(tmp_path):
    calls = []

    def fetch(theme, kind, cols, bbox, release, root):
        calls.append((theme, kind))
        return pd.DataFrame({"id": ["x"], "geometry": [b""]})

    a = load_overture("segment", BBOX, cache_dir=tmp_path, fetch=fetch)
    b = load_overture("segment", BBOX, cache_dir=tmp_path, fetch=fetch)
    assert calls == [("transportation", "segment")] and a["id"].tolist() == b["id"].tolist() == ["x"]
