"""Test tensor."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.matrix_ops.tensor import _kron
from toqito.states import basis

e_0, e_1 = basis(2, 0), basis(2, 1)
matrix1 = np.array([[1, 2]])
matrix2 = np.array([[3], [4]])
matrix3 = np.array([[5, 6]])
matrix4 = np.array([[7, 8]])


@pytest.mark.parametrize(
    "test_input, len_input, expected",
    [
        # standard tensor product on vectors
        ((e_0, e_0), 2, np.kron(e_0, e_0)),
        # tensor product of 1 item to should return the item
        ([np.array([[1, 2], [3, 4]])], 1, np.array([[1, 2], [3, 4]])),
        # tensor product of multiple args as input
        ((np.identity(2), np.identity(2), np.identity(2), np.identity(2)), 4, np.identity(16)),
        # tensor product of array of 2 arrays
        (
            np.array([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]),
            1,
            np.array([[5, 6, 10, 12], [7, 8, 14, 16], [15, 18, 20, 24], [21, 24, 28, 32]]),
        ),
        # tensor product of vector with n = 0
        ((e_0, 0), 2, np.array([[1.0]])),
        # tensor product of vector with n = 1
        ((e_0, 1), 2, e_0),
        # tensor product of vector with n = 2
        ((e_0, 2), 2, np.kron(e_0, e_0)),
        # tensor product of vector with n = 3
        ((e_0, 3), 2, np.kron(np.kron(e_0, e_0), e_0)),
        # tensor product of vector with n = 3
        ((e_0, 4), 2, np.kron(np.kron(np.kron(e_0, e_0), e_0), e_0)),
        # tensor product of list with one item
        ([e_0], 1, e_0),
        # tensor product of list with two items
        ([e_0, e_1], 1, np.kron(e_0, e_1)),
        # tensor product of list with three items
        ([e_0, e_1, e_0], 1, np.kron(np.kron(e_0, e_1), e_0)),
        # tensor product of array of 3 arrays of identity matrices
        (np.array([np.identity(2), np.identity(2), np.identity(2)]), 1, np.identity(8)),
        # ((np.array([np.identity(2), np.identity(2), np.identity(2)])), 1, np.identity(8)),
        # tensor product of array of 4 arrays of identity matrices
        (np.array([np.identity(2), np.identity(2), np.identity(2), np.identity(2)]), 1, np.identity(16)),
        # tensor product with a numpy array containing three or more matrices
        (
            np.array([matrix1, matrix2, matrix3, matrix4], dtype=object),
            1,
            np.kron(np.kron(matrix1, np.kron(matrix2, matrix3)), matrix4),
        ),
        # tensor product of 1 matrix inside a list
        ([np.array([np.identity(4)])], 1, np.identity(4)),
    ],
)
def test_tensor_multiple_input(test_input, len_input, expected):
    """Test function works as expected."""
    if len_input == 1:
        calculated = tensor(test_input)
        assert calculated is expected or (calculated == expected).all()
    elif len_input == 2:
        calculated = tensor(test_input[0], test_input[1])
        assert calculated is expected or (calculated == expected).all()
    elif len_input == 4:
        calculated = tensor(test_input[0], test_input[1], test_input[2], test_input[3])
        assert (calculated == expected).all()


def test_tensor_empty_args():
    r"""Test tensor with no arguments."""
    with pytest.raises(ValueError, match="The `tensor` function must take either a matrix or vector."):
        tensor()


def test_tensor_empty_list():
    r"""An empty input list raises ValueError rather than silently returning None."""
    with pytest.raises(ValueError, match="at least one matrix"):
        tensor([])


@pytest.mark.parametrize(
    "mat_a",
    [
        np.array(3.0),
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.arange(8, dtype=np.complex128).reshape(2, 2, 2),
    ],
)
@pytest.mark.parametrize(
    "mat_b",
    [
        np.array(2.0 + 1j),
        np.array([4.0, 5.0]),
        np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 1.0]]),
        np.arange(12, dtype=np.float64).reshape(1, 3, 4),
    ],
)
def test_kron_matches_numpy_on_mixed_dimensions(mat_a, mat_b):
    r"""`_kron` reimplements NumPy's rule for padding the lower-dimensional operand.

    Every other case delegates to `numpy.kron`, so this is the one behaviour that could
    drift if NumPy ever changed how `kron` broadcasts. Compare bitwise, not approximately.
    """
    expected, actual = np.kron(mat_a, mat_b), _kron(mat_a, mat_b)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")
@pytest.mark.parametrize(
    "mat_a",
    [
        np.matrix([[1.0, 2.0], [3.0, 4.0]]),
        np.ma.masked_array([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]]),
    ],
)
def test_kron_preserves_ndarray_subclasses(mat_a):
    r"""Subclasses route to `numpy.kron` so that its `subok` wrapping is preserved."""
    plain = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert type(_kron(mat_a, plain)) is type(np.kron(mat_a, plain))
