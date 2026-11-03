"""Test generalized_horodecki."""

import numpy as np
import pytest

from toqito.matrix_props import is_density
from toqito.state_props import is_ppt, is_separable
from toqito.states import generalized_horodecki, horodecki


@pytest.mark.parametrize("dim, b_param", [(4, 0.5), (5, 0.3)])
def test_generalized_horodecki_bound_entangled(dim, b_param):
    """For d >= 4 and 0 < b < 1 the state is PPT and certified entangled."""
    rho = generalized_horodecki(dim, b_param)
    assert rho.shape == (2 * dim, 2 * dim)
    assert is_density(rho)
    assert is_ppt(rho, dim=[2, dim])
    separable, witness = is_separable(rho, dim=[2, dim])
    assert not separable
    assert "inconclusive" not in witness


@pytest.mark.parametrize("dim, b_param", [(2, 0.5), (3, 0.5), (6, 0.0), (6, 1.0)])
def test_generalized_horodecki_ppt(dim, b_param):
    """The state is PPT for every dimension and parameter."""
    rho = generalized_horodecki(dim, b_param)
    assert is_density(rho)
    assert is_ppt(rho, dim=[2, dim])


@pytest.mark.parametrize("dim, b_param", [(2, 0.5), (3, 0.5), (4, 0.0)])
def test_generalized_horodecki_separable(dim, b_param):
    """For d = 2, 3 and at b = 0 the state is separable."""
    assert is_separable(generalized_horodecki(dim, b_param), dim=[2, dim])[0]


@pytest.mark.parametrize("dim", [4, 5])
def test_generalized_horodecki_b_one_product_decomposition(dim):
    """At b = 1 the state is the equal mixture of d + 1 product states, so it is separable."""
    rho = np.zeros((2 * dim, 2 * dim), dtype=complex)
    for k in range(dim + 1):
        phase = np.exp(2j * np.pi * k / (dim + 1))
        vec_a = np.array([[phase], [1]]) / np.sqrt(2)
        vec_b = phase ** np.arange(dim).reshape(-1, 1) / np.sqrt(dim)
        vec = np.kron(vec_a, vec_b)
        rho += vec @ vec.conj().T / (dim + 1)
    np.testing.assert_allclose(rho, generalized_horodecki(dim, 1.0), atol=1e-12)


@pytest.mark.parametrize("b_param", [0.2, 0.5, 0.9])
def test_generalized_horodecki_matches_horodecki(b_param):
    """At d = 4 the state is the 2 x 4 Horodecki state with the order of the product basis reversed."""
    reverse = np.kron(np.fliplr(np.identity(2)), np.fliplr(np.identity(4)))
    np.testing.assert_allclose(
        generalized_horodecki(4, b_param), reverse @ horodecki(b_param, [2, 4]) @ reverse.T, atol=1e-12
    )


@pytest.mark.parametrize("dim, b_param", [(1, 0.5), (4, -0.1), (4, 1.1)])
def test_generalized_horodecki_invalid(dim, b_param):
    """Invalid dimensions and parameters are rejected."""
    with pytest.raises(ValueError):
        generalized_horodecki(dim, b_param)
