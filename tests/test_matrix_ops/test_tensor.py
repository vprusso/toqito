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


def test_tensor_empty_args():
    r"""Test tensor with no arguments."""
    with np.testing.assert_raises(ValueError):
        tensor()


if __name__ == "__main__":
    np.testing.run_module_suite()
