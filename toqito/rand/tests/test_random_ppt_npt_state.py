"""Test random_ppt_state, random_npt_state and random_ppt_boundary_state."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.rand import random_npt_state, random_ppt_boundary_state, random_ppt_state
from toqito.state_props import is_ppt


@pytest.mark.parametrize("dims", [[2, 2], [2, 3]])
def test_random_ppt_state(dims):
    """A PPT draw is a density matrix with positive partial transpose."""
    rho = random_ppt_state(dims, seed=0)
    assert rho.shape == (dims[0] * dims[1],) * 2
    assert is_density(rho)
    assert is_ppt(rho, dim=dims)


@pytest.mark.parametrize("dims", [[2, 2], [3, 3]])
def test_random_npt_state(dims):
    """An NPT draw is a density matrix with a negative partial-transpose eigenvalue."""
    rho = random_npt_state(dims, seed=0)
    assert is_density(rho)
    assert not is_ppt(rho, dim=dims)
    assert np.linalg.eigvalsh(partial_transpose(rho, [1], dims)).min() < 0


@pytest.mark.parametrize("func", [random_ppt_state, random_npt_state, random_ppt_boundary_state])
def test_seed_reproducible(func):
    """The same seed gives the same state and different seeds give different states."""
    np.testing.assert_array_equal(func([2, 2], seed=7), func([2, 2], seed=7))
    assert not np.allclose(func([2, 2], seed=7), func([2, 2], seed=8))


@pytest.mark.parametrize("dims", [[2, 2], [2, 3]])
def test_random_ppt_boundary_state(dims):
    """The boundary state is PPT with a partial-transpose eigenvalue within 2^-20 of zero."""
    rho = random_ppt_boundary_state(dims, num_steps=20, seed=3)
    assert is_density(rho)
    assert is_ppt(rho, dim=dims)
    min_eig = np.linalg.eigvalsh(partial_transpose(rho, [1], dims)).min()
    assert -1e-12 < min_eig < 1e-5


def test_random_ppt_boundary_state_zero_steps():
    """With no bisection steps the PPT endpoint itself is returned."""
    rho = random_ppt_boundary_state([2, 2], num_steps=0, seed=1)
    np.testing.assert_allclose(rho, random_ppt_state([2, 2], seed=1))


@pytest.mark.parametrize("dims", [[2], [2, 2, 2], [0, 2], [2.5, 2]])
def test_invalid_dims(dims):
    """Dimensions that are not two positive integers are rejected."""
    with pytest.raises(ValueError):
        random_ppt_state(dims)
    with pytest.raises(ValueError):
        random_npt_state(dims)
    with pytest.raises(ValueError):
        random_ppt_boundary_state(dims)


def test_negative_num_steps():
    """A negative number of bisection steps is rejected."""
    with pytest.raises(ValueError):
        random_ppt_boundary_state([2, 2], num_steps=-1)


def test_max_draws_exhausted():
    """Giving up after max_draws raises instead of looping."""
    with pytest.raises(RuntimeError, match="PPT"):
        random_ppt_state([3, 3], max_draws=5, seed=0)
