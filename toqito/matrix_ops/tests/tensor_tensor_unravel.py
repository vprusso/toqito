import numpy as np
from toqito.matrix_ops.tensor_unravel import tensor_unravel


def test_tensor_unravel_basic():
    """
    2D tensor with one positive element at (1,1).
    Should return [1, 1, 1].
    """
    tensor_constraint = np.array([[-1, -1], [-1, 1]])
    expected = np.array([1, 1, 1])
    result = tensor_unravel(tensor_constraint)
    assert np.array_equal(result, expected)


def test_tensor_unravel_diagonal_unique():
    """
    3D tensor with the unique positive element at (1,1,1).
    Should return [1, 1, 1, 1].
    """
    tensor_constraint = np.full((2, 2, 2), -1)
    tensor_constraint[1, 1, 1] = 1
    expected = np.array([1, 1, 1, 1])
    result = tensor_unravel(tensor_constraint)
    assert np.array_equal(result, expected)


def test_tensor_unravel_invalid_tensor():
    """
    Raise ValueError if no unique positive element exists.
    """
    tensor_constraint = np.full((2, 2), -1)
    try:
        tensor_unravel(tensor_constraint)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "exactly two distinct values" in str(e)


def test_tensor_unravel_multiple_unique_elements():
    """
    Raise ValueError if multiple unique positive elements exist.
    """
    tensor_constraint = np.full((2, 2, 2), -1)
    tensor_constraint[0, 0, 0] = 1
    tensor_constraint[1, 1, 1] = 1
    try:
        tensor_unravel(tensor_constraint)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "unique element" in str(e)
