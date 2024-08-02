"""Tests for left stochastic matrix."""


import numpy as np
import pytest

from toqito.matrices import cyclic_permutation_matrix, pauli
from toqito.matrix_props import is_left_stochastic, is_nonnegative, is_square


@pytest.mark.parametrize("test_input", [(np.identity(3)), (cyclic_permutation_matrix(4)), (pauli("X"))])
def test_true(test_input):
    """Check if function identifies right stochastic matrix correctly."""
    assert is_square(test_input)
    assert is_nonnegative(test_input)
    assert is_left_stochastic(test_input)



@pytest.mark.parametrize("test_input", [(pauli("Y")), (pauli("Z"))])
def test_true(test_input):
    """Check if function identifies non-right stochastic matrix correctly."""
    assert not is_left_stochastic(test_input)
