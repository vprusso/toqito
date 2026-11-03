"""Test grid_state and generalized_grid_state."""

import numpy as np
import pytest

from toqito.matrix_props import is_density
from toqito.state_props import is_ppt, is_separable
from toqito.states import bell, generalized_grid_state, grid_state

PATH_2X3 = [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (1, 0))]


def test_grid_state_single_edge_is_bell_like():
    """One edge gives the pure state (|ij> - |kl>)/sqrt(2)."""
    rho = grid_state([2, 2], [((0, 0), (1, 1))])
    vec = np.array([[1], [0], [0], [-1]]) / np.sqrt(2)
    np.testing.assert_allclose(rho, vec @ vec.T)


def test_grid_state_path_2x3_is_separable():
    """A path on the 2x3 grid is a PPT state of total dimension 6, hence separable."""
    rho = grid_state([2, 3], PATH_2X3)
    assert is_density(rho)
    assert np.linalg.matrix_rank(rho) == 5
    assert is_ppt(rho, dim=[2, 3])
    assert is_separable(rho, dim=[2, 3])[0]


def test_grid_state_uniform_weights():
    """Edges are weighted uniformly, so the trace of each edge projector is 1/|E|."""
    rho = grid_state([3, 3], [((0, 0), (1, 1)), ((0, 1), (1, 2)), ((2, 0), (2, 2))])
    for first, second in [((0, 0), (1, 1)), ((0, 1), (1, 2)), ((2, 0), (2, 2))]:
        vec = np.zeros((9, 1))
        vec[first[0] * 3 + first[1]], vec[second[0] * 3 + second[1]] = 1 / np.sqrt(2), -1 / np.sqrt(2)
        np.testing.assert_allclose(vec.T @ rho @ vec, 1 / 3)


@pytest.mark.parametrize("edges", [[], [((0, 0), (2, 0))], [((0, 0), (0, 3))]])
def test_grid_state_invalid(edges):
    """Empty edge sets and out-of-range vertices are rejected."""
    with pytest.raises(ValueError):
        grid_state([2, 3], edges)


def test_generalized_grid_state_bell():
    """A single two-vertex hyperedge on 2x2 is the Bell state |00> + |11>, which is NPT."""
    rho = generalized_grid_state([2, 2], [[(0, 0), (1, 1)]])
    np.testing.assert_allclose(rho, bell(0) @ bell(0).conj().T)
    assert not is_ppt(rho, dim=[2, 2])


def test_generalized_grid_state_repeated_hyperedge_weight():
    """Listing a hyperedge twice doubles its weight."""
    once = generalized_grid_state([2, 2], [[(0, 0)], [(1, 1)]])
    twice = generalized_grid_state([2, 2], [[(0, 0)], [(0, 0)], [(1, 1)]])
    np.testing.assert_allclose(np.diag(once), [0.5, 0, 0, 0.5])
    np.testing.assert_allclose(np.diag(twice), [2 / 3, 0, 0, 1 / 3])


@pytest.mark.parametrize("hyperedges", [[], [[]], [[(0, 2)]]])
def test_generalized_grid_state_invalid(hyperedges):
    """Empty hyperedge sets, empty hyperedges and out-of-range vertices are rejected."""
    with pytest.raises(ValueError):
        generalized_grid_state([2, 2], hyperedges)
