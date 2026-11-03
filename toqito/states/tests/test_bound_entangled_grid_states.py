"""Test the named bound entangled states cross_hatch, krebs_gachechiladze, vertesi_brunner, moroder_steering, piani."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.state_props import is_ppt, is_separable
from toqito.states import cross_hatch, krebs_gachechiladze, moroder_steering, piani, vertesi_brunner


@pytest.mark.parametrize(
    "rho, dims, rank",
    [
        (cross_hatch(), [3, 3], 4),
        (krebs_gachechiladze(), [5, 5], 11),
        (vertesi_brunner(), [3, 3], 4),
        (moroder_steering(0.3, 0.4), [3, 3], 4),
        (moroder_steering(0.1, 0.5), [3, 3], 4),
        (piani(), [4, 4], 6),
    ],
)
def test_bound_entangled(rho, dims, rank):
    """Each state is a density operator, PPT, of the expected rank, and certified entangled by realignment."""
    assert is_density(rho)
    assert np.linalg.matrix_rank(rho) == rank
    assert is_ppt(rho, dim=dims)
    separable, witness = is_separable(rho, dim=dims)
    assert not separable
    assert "realignment" in witness


@pytest.mark.parametrize("m1, m2", [(0.3, 0.4), (0.0, 0.0), (1 / 60, 3 / 10)])
def test_moroder_steering_partial_transpose_invariant(m1, m2):
    """The Moroder family is invariant under partial transposition, which is how the weights are fixed."""
    rho = moroder_steering(m1, m2)
    np.testing.assert_allclose(partial_transpose(rho, [1], [3, 3]), rho, atol=1e-12)


def test_cross_hatch_not_partial_transpose_invariant():
    """The cross-hatch state is PPT but not invariant under partial transposition."""
    rho = cross_hatch()
    assert not np.allclose(partial_transpose(rho, [1], [3, 3]), rho)


def test_vertesi_brunner_spectrum():
    """The non-zero eigenvalues are 3257/6884, 450/1721 (twice), 27/6884."""
    eigs = np.sort(np.linalg.eigvalsh(vertesi_brunner()))[::-1]
    np.testing.assert_allclose(eigs[:4], [3257 / 6884, 450 / 1721, 450 / 1721, 27 / 6884], atol=1e-12)
    np.testing.assert_allclose(eigs[4:], 0, atol=1e-12)


def test_vertesi_brunner_shares_moroder_spectrum():
    """At m1 = 1/60, m2 = 3/10 the Moroder state has the Vertesi-Brunner spectrum but is a different matrix."""
    vb, ms = vertesi_brunner(), moroder_steering(1 / 60, 3 / 10)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(vb)), np.sort(np.linalg.eigvalsh(ms)), atol=1e-12)
    assert not np.allclose(vb, ms)


@pytest.mark.parametrize("m1, m2", [(0.6, 0.6), (-0.1, 0.2), (0.2, -0.1), (0.0, 1.01)])
def test_moroder_steering_invalid(m1, m2):
    """Parameters outside m1, m2 >= 0 and m1^2 + m2^2 + m1 m2 <= 1 are rejected."""
    with pytest.raises(ValueError):
        moroder_steering(m1, m2)


def test_moroder_steering_boundary_is_density():
    """On the boundary m1^2 + m2^2 + m1 m2 = 1 the weights are still non-negative."""
    m1 = 0.5
    m2 = (-m1 + np.sqrt(m1**2 - 4 * (m1**2 - 1))) / 2
    assert is_density(moroder_steering(m1, m2))


def test_piani_hermitian_real():
    """The Piani state is real symmetric despite the complex Pauli Y factors."""
    rho = piani()
    np.testing.assert_allclose(rho.imag, 0, atol=1e-12)
    np.testing.assert_allclose(rho, rho.T, atol=1e-12)
