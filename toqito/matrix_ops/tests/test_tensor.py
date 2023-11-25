"""Test tensor."""
import numpy as np

from toqito.matrix_ops import tensor
from toqito.states import basis


def test_tensor():
    """Test standard tensor on vectors."""
    e_0 = basis(2, 0)
    expected_res = np.kron(e_0, e_0)

    res = tensor(e_0, e_0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_single_arg():
    """Performing tensor product on one item should return item back."""
    input_arr = np.array([[1, 2], [3, 4]])
    res = tensor(input_arr)

    bool_mat = np.isclose(res, input_arr)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_array_of_numpy_arrays_two():
    """Performing tensor product on two numpy array of numpy arrays."""
    input_arr = np.array([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])])
    res = tensor(input_arr)

    expected_res = np.array([[5, 6, 10, 12], [7, 8, 14, 16], [15, 18, 20, 24], [21, 24, 28, 32]])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_array_of_numpy_arrays_three():
    """Performing tensor product on three numpy array of numpy arrays."""
    input_arr = np.array([np.identity(2), np.identity(2), np.identity(2)])
    res = tensor(input_arr)

    expected_res = np.identity(8)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_array_of_numpy_arrays_four():
    """Performing tensor product on four numpy array of numpy arrays."""
    input_arr = np.array([np.identity(2), np.identity(2), np.identity(2), np.identity(2)])
    res = tensor(input_arr)

    expected_res = np.identity(16)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_multiple_args():
    """Performing tensor product on multiple matrices."""
    input_arr_1 = np.identity(2)
    input_arr_2 = np.identity(2)
    input_arr_3 = np.identity(2)
    input_arr_4 = np.identity(2)
    res = tensor(input_arr_1, input_arr_2, input_arr_3, input_arr_4)

    bool_mat = np.isclose(res, np.identity(16))
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_n_0():
    """Test tensor n=0 times."""
    e_0 = basis(2, 0)
    expected_res = None

    res = tensor(e_0, 0)
    np.testing.assert_equal(res, expected_res)


def test_tensor_n_1():
    """Test tensor n=1 times."""
    e_0 = basis(2, 0)
    expected_res = e_0

    res = tensor(e_0, 1)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_n_2():
    """Test tensor n=2 times."""
    e_0 = basis(2, 0)
    expected_res = np.kron(e_0, e_0)

    res = tensor(e_0, 2)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_n_3():
    """Test tensor n=3 times."""
    e_0 = basis(2, 0)
    expected_res = np.kron(np.kron(e_0, e_0), e_0)

    res = tensor(e_0, 3)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_list_0():
    """Test tensor empty list."""
    expected_res = None

    res = tensor([])
    np.testing.assert_equal(res, expected_res)


def test_tensor_list_1():
    """Test tensor list with one item."""
    e_0 = basis(2, 0)
    expected_res = e_0

    res = tensor([e_0])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_list_2():
    """Test tensor list with two items."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = np.kron(e_0, e_1)

    res = tensor([e_0, e_1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_list_3():
    """Test tensor list with three items."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = np.kron(np.kron(e_0, e_1), e_0)

    res = tensor([e_0, e_1, e_0])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tensor_with_three_or_more_matrices():
    """Test tensor product with a numpy array containing three or more matrices."""
    # Three matrices to be Kronecker multiplied
    matrix1 = np.array([[1, 2]])
    matrix2 = np.array([[3], [4]])
    matrix3 = np.array([[5, 6]])
    matrix4 = np.array([[7, 8]])

    # The numpy array containing the matrices
    matrices = np.array([matrix1, matrix2, matrix3, matrix4], dtype=object)

    # Expected output: Kronecker product of matrix1, matrix2, and matrix3
    expected_output = np.kron(np.kron(matrix1, np.kron(matrix2, matrix3)), matrix4)

    # Call the tensor function
    result = tensor(matrices)

    # Assert that the result is as expected
    np.testing.assert_array_equal(result, expected_output)


def test_tensor_empty_args():
    r"""Test tensor with no arguments."""
    with pytest.raises(ValueError):
        tensor()