"""Tests for is_antidistinguishable_circulant."""

import numpy as np
import pytest

from toqito.rand import random_circulant_gram_matrix
from toqito.matrix_ops import vectors_from_gram_matrix
from toqito.state_props import is_antidistinguishable, is_antidistinguishable_circulant


def _trine_gram() -> np.ndarray:
    """Gram matrix for the trine states (inner product -1/2 off-diagonal)."""
    return np.array([[1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]], dtype=float)


def _equiangular_gram(n: int, c: float) -> np.ndarray:
    """Circulant Gram matrix: 1 on diagonal, c everywhere else."""
    return (1 - c) * np.eye(n) + c * np.ones((n, n))


@pytest.mark.parametrize("gram, expected", [
    # Trine states: exactly antidistinguishable (gap = 0)
    (_trine_gram(), True),
    # Orthogonal states (identity Gram): all eigenvalues equal -> AD
    (np.eye(3), True),
    # 4 states, very small off-diagonal -> AD
    (_equiangular_gram(4, 0.01), True),
    # 2 identical states: rank-1 Gram -> NOT AD
    (np.array([[1.0, 1.0], [1.0, 1.0]]), False),
])
def test_known_cases(gram, expected):
    is_ad, _ = is_antidistinguishable_circulant(gram)
    assert is_ad == expected


def test_trine_gap_is_zero():
    """For trine states the gap should be exactly 0 (boundary case)."""
    _, gap = is_antidistinguishable_circulant(_trine_gram())
    assert abs(gap) < 1e-6


def test_skip_circulant_check():
    is_ad, _ = is_antidistinguishable_circulant(_trine_gram(), skip_circulant_check=True)
    assert is_ad is True


@pytest.mark.parametrize("seed", range(10))
def test_agrees_with_sdp(seed):
    """Closed-form result must match is_antidistinguishable (SDP) on random circulant states."""
    gram = random_circulant_gram_matrix(5, seed=seed)
    states = vectors_from_gram_matrix(gram)
    is_ad_sdp = is_antidistinguishable(states)
    is_ad_cf, _ = is_antidistinguishable_circulant(gram)
    assert is_ad_cf == is_ad_sdp


def test_non_circulant_raises():
    # Hermitian but not circulant
    non_circ = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.0]])
    with pytest.raises(ValueError, match="not circulant"):
        is_antidistinguishable_circulant(non_circ)


def test_non_square_raises():
    with pytest.raises(ValueError, match="square"):
        is_antidistinguishable_circulant(np.ones((2, 3)))


def test_non_hermitian_raises():
    m = np.array([[1.0, 0.5 + 0.1j], [0.5 - 0.2j, 1.0]])
    with pytest.raises(ValueError, match="Hermitian"):
        is_antidistinguishable_circulant(m)