"""Test random_unitary."""

import pytest

from toqito.matrix_props import is_unitary
from toqito.rand import random_unitary


@pytest.mark.parametrize("dim", range(2, 8))
@pytest.mark.parametrize("is_real", [True, False])
def test_random_unitary_int_dim(dim, is_real):
    """Test function works as expected for a valid int input."""
    mat = random_unitary(dim=dim, is_real=is_real)
    assert is_unitary(mat)


@pytest.mark.parametrize("dim_n", range(2, 8))
@pytest.mark.parametrize("dim_m", range(2, 8))
@pytest.mark.parametrize("is_real", [True, False])
def test_random_unitary_list_dims(dim_n, dim_m, is_real):
    """Test function works as expected for a valid input of list."""
    if dim_n == dim_m:
        mat = random_unitary(dim=[dim_n, dim_m], is_real=is_real)
        assert is_unitary(mat)


@pytest.mark.parametrize("dim_n", range(2, 8))
@pytest.mark.parametrize("dim_m", range(2, 8))
@pytest.mark.parametrize("is_real", [True, False])
def test_non_square_dims(dim_n, dim_m, is_real):
    """Unitary matrices must be square are not allowed."""
    if dim_n != dim_m:
        with pytest.raises(ValueError, match="Unitary matrix must be square."):
            random_unitary(dim=[dim_n, dim_m], is_real=is_real)
