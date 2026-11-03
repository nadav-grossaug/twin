import numpy as np
import pytest

from graph_aim.transfer.binning import check_bin_edges, degree_bin


def test_degree_bin_assigns_half_open_intervals():
    edges = [0, 2, 6, 16]
    deg = np.array([0, 1, 2, 5, 6, 15, 16, 1000])
    assert degree_bin(deg, edges).tolist() == [0, 0, 1, 1, 2, 2, 2, 2]  # last bin is open-ended


def test_degree_bin_single_bin():
    assert degree_bin(np.array([0, 3, 99]), [0, 5]).tolist() == [0, 0, 0]


def test_degree_bin_empty_input():
    assert degree_bin(np.array([], dtype=np.int64), [0, 1, 2]).shape == (0,)


@pytest.mark.parametrize("bad", [[0], [], [1, 2, 3], [0, 3, 3], [0, 5, 2], [[0, 1], [2, 3]], [0, 1.5, 3]])
def test_check_bin_edges_rejects(bad):
    with pytest.raises(ValueError):
        check_bin_edges(bad)


def test_check_bin_edges_accepts_integer_valued_floats():
    assert check_bin_edges([0.0, 2.0, 5.0]).tolist() == [0, 2, 5]
