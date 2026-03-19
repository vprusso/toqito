"""Tests for random_superoperator."""

import numpy as np
import pytest

from toqito.channel_props import is_quantum_channel
from toqito.matrix_props import is_positive_semidefinite
from toqito.rand import random_superoperator


@pytest.mark.parametrize(
    "dim, is_real",
    [
        (2, False),
        (2, True),
        (3, False),
        (4, True),
    ],
)
def test_tp_channel_is_valid(dim, is_real):
    """Trace-preserving random channel should be a valid quantum channel."""
    choi = random_superoperator(dim, is_real=is_real, seed=0)
    assert choi.shape == (dim**2, dim**2)
    assert is_positive_semidefinite(choi)
    assert is_quantum_channel(choi)


@pytest.mark.parametrize("dim", [2, 3])
def test_cp_only_is_psd(dim):
    """CP-only (not TP) channel should produce a PSD Choi matrix."""
    choi = random_superoperator(dim, is_trace_preserving=False, seed=0)
    assert choi.shape == (dim**2, dim**2)
    assert is_positive_semidefinite(choi)


def test_rectangular_dimensions():
    """Channel from 2-dim to 3-dim system."""
    choi = random_superoperator([2, 3], seed=0)
    assert choi.shape == (6, 6)
    assert is_positive_semidefinite(choi)


def test_seed_reproducibility():
    """Same seed should produce the same Choi matrix."""
    a = random_superoperator(3, seed=42)
    b = random_superoperator(3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    """Different seeds should produce different results."""
    a = random_superoperator(3, seed=1)
    b = random_superoperator(3, seed=2)
    assert not np.array_equal(a, b)


def test_real_channel_is_real():
    """Real channel should have no imaginary part."""
    choi = random_superoperator(2, is_real=True, seed=0)
    assert np.allclose(choi.imag, 0)


def test_invalid_dim_raises():
    """Invalid dimension should raise ValueError."""
    with pytest.raises(ValueError):
        random_superoperator([2, 3, 4])


def test_zero_dim_raises():
    """Zero dimension should raise ValueError."""
    with pytest.raises(ValueError):
        random_superoperator(0)
