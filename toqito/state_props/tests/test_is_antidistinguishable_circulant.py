"""Test is_antidistinguishable_circulant."""

import numpy as np
import pytest

from toqito.matrix_ops import vectors_from_gram_matrix, vectors_to_gram_matrix
from toqito.rand import random_circulant_gram_matrix
from toqito.state_props import is_antidistinguishable, is_antidistinguishable_circulant
from toqito.states import trine


@pytest.mark.parametrize("skip_circulant_check", [False, True])
def test_is_antidistinguishable_circulant_trine(skip_circulant_check):
    """Test function works as expected for a valid set of states."""
    assert is_antidistinguishable_circulant(trine(), skip_circulant_check=skip_circulant_check)


def test_is_antidistinguishable_circulant_complex_gram():
    """Test that the circulant set of states |+>, |i>, |-> and |-i> are antidistinguishable."""
    gram_matrix = np.array(
        [
            [1, (1 + 1j) / 2, 0, (1 - 1j) / 2],
            [(1 - 1j) / 2, 1, (1 + 1j) / 2, 0],
            [0, (1 - 1j) / 2, 1, (1 + 1j) / 2],
            [(1 + 1j) / 2, 0, (1 - 1j) / 2, 1],
        ]
    )
    assert is_antidistinguishable_circulant(gram_matrix)


def test_is_antidistinguishable_circulant_trine_gram():
    """Test function works as expected for a valid set of states."""
    assert is_antidistinguishable_circulant(vectors_to_gram_matrix(trine()))


@pytest.mark.parametrize("dim", [3, 5, 10])
def test_is_antidistinguishable_random_circulant_gram(dim):
    """Test function works as expected for random circulant Gram matrices."""
    for i in range(5):
        gram_matrix = random_circulant_gram_matrix(dim, seed=10 * dim + i)
        states = vectors_from_gram_matrix(gram_matrix)
        assert is_antidistinguishable(states) == is_antidistinguishable_circulant(states)


@pytest.mark.parametrize("n", [2, 3, 10])
def test_is_antidistinguishable_circulant_gram_closed_form(n):
    """Test function works as expected for circulant Gram matrices with closed-form criterion."""
    threshold = (n - 2) / (n - 1)
    eps = 0.2

    for gamma, antidistinguishable in [(threshold, True), (threshold + eps, False)]:
        gram_matrix = gamma * np.ones((n, n)) + (1 - gamma) * np.eye(n)
        assert is_antidistinguishable_circulant(gram_matrix) == antidistinguishable


def test_is_antidistinguishable_circulant_not_circulant():
    """Test function raises ValueError for not circulant Gram matrices."""
    gram_matrix = np.arange(9).reshape(3, 3).astype(float)

    with pytest.raises(ValueError):
        is_antidistinguishable_circulant(gram_matrix)
