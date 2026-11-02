from geolife.eval.plots import plot_artifact
from geolife.transfer.artifact import load_artifact
from tests.geolife.transfer.test_artifact import build


def test_plots(tmp_path):
    build(tmp_path / "art")
    pngs = plot_artifact(load_artifact(tmp_path / "art"), tmp_path / "png")
    assert len(pngs) == 2 and all(p.stat().st_size > 0 for p in pngs)
