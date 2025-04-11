"""Tests for right, left and doubly stochastic matrix."""

import numpy as np
import pytest

from toqito.matrices import cyclic_permutation_matrix, pauli
from toqito.matrix_props import is_stochastic


@pytest.mark.parametrize("matrix", [np.eye(3), cyclic_permutation_matrix(4), pauli("X")])
@pytest.mark.parametrize("mat_type", ["left", "right", "doubly"])
def test_valid_stochastic_matrices(matrix, mat_type):
    """Valid stochastic matrices should return True for all valid types."""
    assert is_stochastic(matrix, mat_type)


@pytest.mark.parametrize("matrix", [pauli("Y"), pauli("Z")])
@pytest.mark.parametrize("mat_type", ["left", "right", "doubly"])
def test_non_stochastic_matrices(matrix, mat_type):
    """Non-stochastic matrices (with negative or incorrect row/column sums) should return False."""
    assert not is_stochastic(matrix, mat_type)


@pytest.mark.parametrize("matrix", [pauli("Y"), pauli("Z"), pauli("X"), pauli("I")])
@pytest.mark.parametrize("bad_type", ["l", "r", 1, (), "d"])
def test_invalid_stochastic_type_raises(matrix, bad_type):
    """Invalid mat_type values should raise a TypeError."""
    with pytest.raises(TypeError):
        is_stochastic(matrix, bad_type)
