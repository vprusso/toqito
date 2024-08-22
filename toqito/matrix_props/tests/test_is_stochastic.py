"""Tests for right, left and doubly stochastic matrix."""

import numpy as np
import pytest

from toqito.matrices import cyclic_permutation_matrix, pauli
from toqito.matrix_props import is_nonnegative, is_square, is_stochastic


@pytest.mark.parametrize("test_input", [(np.identity(3)), (cyclic_permutation_matrix(4)), (pauli("X"))])
@pytest.mark.parametrize("test_type", [("left"), ("right"), ("doubly")])
def test_true(test_input, test_type):
    """Check if function identifies right stochastic matrix correctly."""
    assert is_stochastic(test_input, test_type)



@pytest.mark.parametrize("test_input", [(pauli("Y")), (pauli("Z"))])
@pytest.mark.parametrize("test_type", [("left"), ("right"), ("doubly")])
def test_true(test_input, test_type):
    """Check if function identifies non-right stochastic matrix correctly."""
    assert not is_stochastic(test_input, test_type)

@pytest.mark.parametrize("test_input", [(pauli("Y")), (pauli("Z")), (pauli("X")), (pauli("I"))])
@pytest.mark.parametrize("bad_type", [("l"), ("r"), (1), (), ("d")])
def test_true(test_input, bad_type):
    """Check if error raised correctly for invalid stochastic matrix type."""
    with pytest.raises(TypeError):
        is_stochastic(test_input, bad_type)
