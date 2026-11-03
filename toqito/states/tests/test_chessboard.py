"""Test chessboard."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.state_props import is_ppt
from toqito.states import chessboard


def test_chessboard():
    """The chessboard_state."""
    res = chessboard([1, 2, 3, 4, 5, 6], 7, 8)
    np.testing.assert_allclose(res[0][0], 0.22592592592592592)


def test_chessboard_default_s():
    """The chessboard_state with default `s_param`."""
    res = chessboard([1, 2, 3, 4, 5, 6], 7)
    np.testing.assert_allclose(res[0][0], 0.29519938056523426)


def test_chessboard_default_s_t():
    """The chessboard_state with default `s_param` and `t_param`."""
    res = chessboard([1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(res[0][0], 0.3863449236810438)


@pytest.mark.parametrize("seed", range(5))
def test_chessboard_default_is_ppt(seed):
    """With the Bruss-Peres defaults s = ac/n and t = ad/m the state is invariant under partial transposition."""
    rng = np.random.default_rng(seed)
    rho = chessboard(list(rng.normal(size=6)))
    assert is_density(rho)
    np.testing.assert_allclose(partial_transpose(rho, [1], [3, 3]), rho, atol=1e-12)
    assert is_ppt(rho, dim=[3, 3])


def test_chessboard_other_s_is_npt():
    """Dropping the factor a from s gives a state with negative partial transpose."""
    rng = np.random.default_rng(7)
    params = list(rng.normal(size=6))
    a_param, _, c_param, _, _, n_param = params
    assert not is_ppt(chessboard(params, c_param / n_param), dim=[3, 3])


def test_chessboard_extremal_ppt_state():
    """The extremal PPT entangled member of Imai et al. is PPT and has rank 4."""
    rho = chessboard([3 / 5, -3 / 5, 6 / 5, -6 / 5, -3 / 5, -3 / 5])
    assert is_ppt(rho, dim=[3, 3])
    assert np.linalg.matrix_rank(rho) == 4
