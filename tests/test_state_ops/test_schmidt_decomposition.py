"""Test schmidt_decomposition."""
import numpy as np

from toqito.state_ops import schmidt_decomposition
from toqito.states import basis, max_entangled


def test_schmidt_decomp_max_ent():
    """Schmidt decomposition of the 3-D maximally entangled state."""
    singular_vals, u_mat, vt_mat = schmidt_decomposition(max_entangled(3))

    expected_u_mat = np.identity(3)
    expected_vt_mat = np.identity(3)
    expected_singular_vals = 1 / np.sqrt(3) * np.array([[1], [1], [1]])

    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_schmidt_decomp_two_qubit_1():
    """
    Schmidt decomposition of two-qubit state.

    The Schmidt decomposition of | phi > = 1/2(|00> + |01> + |10> + |11>) is
    the state |+>|+> where |+> = 1/sqrt(2) * (|0> + |1>).
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)

    phi = 1 / 2 * (np.kron(e_0, e_0) + np.kron(e_0, e_1) + np.kron(e_1, e_0) + np.kron(e_1, e_1))
    singular_vals, vt_mat, u_mat = schmidt_decomposition(phi)

    expected_singular_vals = np.array([[1]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = 1 / np.sqrt(2) * np.array([[-1], [-1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = 1 / np.sqrt(2) * np.array([[-1], [-1]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_schmidt_decomp_two_qubit_2():
    """
    Schmidt decomposition of two-qubit state.

    The Schmidt decomposition of | phi > = 1/2(|00> + |01> + |10> - |11>) is
    the state 1/sqrt(2) * (|0>|+> + |1>|->).
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)

    phi = 1 / 2 * (np.kron(e_0, e_0) + np.kron(e_0, e_1) + np.kron(e_1, e_0) - np.kron(e_1, e_1))
    singular_vals, vt_mat, u_mat = schmidt_decomposition(phi)

    expected_singular_vals = 1 / np.sqrt(2) * np.array([[1], [1]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = np.array([[-1, 0], [0, -1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = 1 / np.sqrt(2) * np.array([[-1, -1], [-1, 1]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    s_decomp = (
        singular_vals[0] * np.atleast_2d(np.kron(vt_mat[:, 0], u_mat[:, 0])).T
        + singular_vals[1] * np.atleast_2d(np.kron(vt_mat[:, 1], u_mat[:, 1])).T
    )

    np.testing.assert_equal(np.isclose(np.linalg.norm(phi - s_decomp), 0), True)


def test_schmidt_decomp_two_qubit_3():
    """
    Schmidt decomposition of two-qubit state.

    The Schmidt decomposition of 1/2* (|00> + |11>) has Schmidt coefficients
    equal to 1/2[1, 1]
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)

    phi = 1 / 2 * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    singular_vals, vt_mat, u_mat = schmidt_decomposition(phi)

    expected_singular_vals = 1 / 2 * np.array([[1], [1]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = np.array([[1, 0], [0, 1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = np.array([[1, 0], [0, 1]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    s_decomp = (
        singular_vals[0] * np.atleast_2d(np.kron(vt_mat[:, 0], u_mat[:, 0])).T
        + singular_vals[1] * np.atleast_2d(np.kron(vt_mat[:, 1], u_mat[:, 1])).T
    )

    np.testing.assert_equal(np.isclose(np.linalg.norm(phi - s_decomp), 0), True)


def test_schmidt_decomp_two_qubit_4():
    """
    Schmidt decomposition of two-qubit state.

    The Schmidt decomposition of 1/2 * (|00> - |01> + |10> + |11>) has Schmidt coefficients
    equal to [1, 1]
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)

    phi = 1 / 2 * (np.kron(e_0, e_0) - np.kron(e_0, e_1) + np.kron(e_1, e_0) + np.kron(e_1, e_1))
    singular_vals, vt_mat, u_mat = schmidt_decomposition(phi)

    expected_singular_vals = 1 / np.sqrt(2) * np.array([[1], [1]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = np.array([[-1, 0], [0, 1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_schmidt_decomp_dim_list():
    """Schmidt decomposition with list specifying dimension."""
    singular_vals, u_mat, vt_mat = schmidt_decomposition(max_entangled(3), dim=[3, 3])

    expected_u_mat = np.identity(3)
    expected_vt_mat = np.identity(3)
    expected_singular_vals = 1 / np.sqrt(3) * np.array([[1], [1], [1]])

    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_schmidt_decomp_dim_list_pure_state():
    """Schmidt decomposition of a pure state with a dimension list."""
    pure_vec = -1 / np.sqrt(2) * np.array([[1], [0], [1], [0]])

    # Test when dimension default and k_param is default (0):
    singular_vals, vt_mat, u_mat = schmidt_decomposition(pure_vec)

    expected_singular_vals = np.array([[1]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = 1 / np.sqrt(2) * np.array([[-1], [-1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = np.array([[1], [0]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Test when dimension [2, 2] and k_param is 1:
    singular_vals, vt_mat, u_mat = schmidt_decomposition(pure_vec, [2, 2], 1)

    expected_singular_vals = np.array([[1]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = 1 / np.sqrt(2) * np.array([[-1], [-1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = np.array([[1], [0]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Test when dimension [2, 2] and k_param is 2:
    singular_vals, vt_mat, u_mat = schmidt_decomposition(pure_vec, [2, 2], 2)

    expected_singular_vals = np.array([[1], [0]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = 1 / np.sqrt(2) * np.array([[-1, -1], [-1, 1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = np.identity(2)
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_schmidt_decomp_standard_basis():
    """Test on standard basis vectors."""
    e_1 = basis(2, 1)
    singular_vals, vt_mat, u_mat = schmidt_decomposition(np.kron(e_1, e_1))

    expected_singular_vals = np.array([[1]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = np.array([[0], [1]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = np.array([[0], [1]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_schmidt_decomp_example():
    """Test for example Schmidt decomposition."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    phi = (
        (1 + np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_0)
        + (1 - np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_1)
        + (np.sqrt(2) - np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_0)
        + (np.sqrt(2) + np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_1)
    )

    singular_vals, vt_mat, u_mat = schmidt_decomposition(phi)

    expected_singular_vals = np.array([[np.sqrt(3 / 4)], [np.sqrt(1 / 4)]])
    bool_mat = np.isclose(expected_singular_vals, singular_vals)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_vt_mat = np.array([[-0.81649658, 0.57735027], [0.57735027, 0.81649658]])
    bool_mat = np.isclose(expected_vt_mat, vt_mat)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_u_mat = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
    bool_mat = np.isclose(expected_u_mat, u_mat)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
