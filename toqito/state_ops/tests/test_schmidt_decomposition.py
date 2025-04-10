"""Test schmidt_decomposition."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.rand import random_density_matrix, random_state_vector
from toqito.state_ops import schmidt_decomposition
from toqito.states import basis, max_entangled

e_0, e_1 = basis(2, 0), basis(2, 1)
phi1 = 1 / 2 * (np.kron(e_0, e_0) + np.kron(e_0, e_1) + np.kron(e_1, e_0) + np.kron(e_1, e_1))
phi2 = 1 / 2 * (np.kron(e_0, e_0) + np.kron(e_0, e_1) + np.kron(e_1, e_0) - np.kron(e_1, e_1))
phi3 = 1 / 2 * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
phi4 = 1 / 2 * (np.kron(e_0, e_0) - np.kron(e_0, e_1) + np.kron(e_1, e_0) + np.kron(e_1, e_1))
pure_vec = -1 / np.sqrt(2) * np.array([[1], [0], [1], [0]])


@pytest.mark.parametrize(
    "test_input, expected_u_mat, expected_vt_mat, expected_singular_vals, reconstruct",
    [
        # Schmidt decomposition of the 3-D maximally entangled state
        (max_entangled(3), np.identity(3), np.identity(3), 1 / np.sqrt(3) * np.array([[1], [1], [1]]), False),
        # Schmidt decomposition of two-qubit state. The Schmidt decomposition of | phi > = 1/2(|00> + |01> + |10> +
        # |11>) is the state |+>|+> where |+> = 1/sqrt(2) * (|0> + |1>).
        (
            phi1,
            1 / np.sqrt(2) * np.array([[-1], [-1]]),
            1 / np.sqrt(2) * np.array([[-1], [-1]]),
            np.array([[1]]),
            False,
        ),
        # Schmidt decomposition of two-qubit state. The Schmidt decomposition of | phi > = 1/2(|00> + |01> + |10> -
        # |11>) is the state 1/sqrt(2) * (|0>|+> + |1>|->).
        (
            phi2,
            # np.array([[-1, -1], [-1, 1]]),
            np.array([[-1, 0], [0, -1]]),
            1 / np.sqrt(2) * np.array([[-1, -1], [-1, 1]]),
            1 / np.sqrt(2) * np.array([[1], [1]]),
            True,
        ),
        # Schmidt decomposition of two-qubit state. The Schmidt decomposition of 1/2* (|00> + |11>) has Schmidt
        # coefficients equal to 1/2[1, 1]
        (phi3, np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), 1 / 2 * np.array([[1], [1]]), True),
        # Schmidt decomposition of two-qubit state. The Schmidt decomposition of 1/2 * (|00> - |01> + |10> + |11>) has
        # Schmidt coefficients equal to [1, 1]
        (
            phi4,
            np.array([[-1, 0], [0, 1]]),
            1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]]),
            1 / np.sqrt(2) * np.array([[1], [1]]),
            False,
        ),
        # Schmidt decomposition of a pure state with a dimension list
        (pure_vec, 1 / np.sqrt(2) * np.array([[-1], [-1]]), np.array([[1], [0]]), np.array([[1]]), False),
        # Test on standard basis vectors
        (np.kron(e_1, e_1), np.array([[0], [1]]), np.array([[0], [1]]), np.array([[1]]), False),
        # separable density matrix
        (
            np.identity(4),
            np.array([[[-0.70710678], [0.0]], [[0.0], [-0.70710678]]]),
            np.array([[[-0.70710678], [0.0]], [[0.0], [-0.70710678]]]),
            np.array([[2.0]]),
            False,
        ),
    ],
)
def test_schmidt_decomposition_no_input_dim(
    test_input, expected_u_mat, expected_vt_mat, expected_singular_vals, reconstruct
):
    """Test function works as expected for valid inputs."""
    calculated_singular_vals, calculated_u_mat, calculated_vt_mat = schmidt_decomposition(test_input)

    assert np.allclose(calculated_singular_vals, expected_singular_vals)
    assert np.allclose(calculated_u_mat, expected_u_mat)
    assert np.allclose(calculated_vt_mat, expected_vt_mat)

    if reconstruct is True:
        s_decomp = (
            calculated_singular_vals[0] * np.atleast_2d(np.kron(calculated_u_mat[:, 0], calculated_vt_mat[:, 0])).T
            + calculated_singular_vals[1] * np.atleast_2d(np.kron(calculated_u_mat[:, 1], calculated_vt_mat[:, 1])).T
        )
        assert np.linalg.norm(test_input - s_decomp) <= 0.001


@pytest.mark.parametrize(
    "test_input, input_dim, input_param, expected_u_mat, expected_vt_mat, expected_singular_vals",
    [
        # Schmidt decomposition of the 3-D maximally entangled state
        (max_entangled(3), [3, 3], None, np.identity(3), np.identity(3), 1 / np.sqrt(3) * np.array([[1], [1], [1]])),
        # Schmidt decomposition of a pure state with a dimension list when input_dim is list and k_param is 1
        (pure_vec, [2, 2], 1, 1 / np.sqrt(2) * np.array([[-1], [-1]]), np.array([[1], [0]]), np.array([[1]])),
        # Schmidt decomposition of a pure state with a dimension list when input_dim is list and k_param is 2
        (pure_vec, [2, 2], 2, 1 / np.sqrt(2) * np.array([[-1, -1], [-1, 1]]), np.identity(2), np.array([[1], [0]])),
        # Input dim is None - Schmidt decomposition of the 3-D maximally entangled state
        (max_entangled(3), None, None, np.identity(3), np.identity(3), 1 / np.sqrt(3) * np.array([[1], [1], [1]])),
        # separable density matrix
        (
            np.identity(4),
            2,
            None,
            np.array([[[-0.70710678], [0.0]], [[0.0], [-0.70710678]]]),
            np.array([[[-0.70710678], [0.0]], [[0.0], [-0.70710678]]]),
            np.array([[2.0]]),
        ),
    ],
)
def test_schmidt_decomposition_input_dim(
    test_input, input_dim, input_param, expected_u_mat, expected_vt_mat, expected_singular_vals
):
    """Test function works as expected for valid inputs."""
    if input_param is None:
        calculated_singular_vals, calculated_u_mat, calculated_vt_mat = schmidt_decomposition(test_input, dim=input_dim)
    else:
        calculated_singular_vals, calculated_u_mat, calculated_vt_mat = schmidt_decomposition(
            test_input, dim=input_dim, k_param=input_param
        )
    assert (calculated_singular_vals - expected_singular_vals).all() <= 0.1
    assert (calculated_u_mat - expected_u_mat).all() <= 0.1
    assert (calculated_vt_mat - expected_vt_mat).all() <= 0.1


def test_schmidt_decomp_random_state():
    """Test for random state."""
    rho = random_state_vector(8)
    singular_vals, u_mat, vt_mat = schmidt_decomposition(rho, [2, 4])
    reconstructed = np.sum(
        [singular_vals[i, 0] * tensor(u_mat[:, [i]], vt_mat[:, [i]]) for i in range(len(singular_vals))],
        axis=0,
    )
    assert np.isclose(rho, reconstructed).all()


def test_schmidt_decomp_random_operator():
    """Test for random operator."""
    rho = random_density_matrix(8)
    singular_vals, u_mat, vt_mat = schmidt_decomposition(rho, [2, 4])
    reconstructed = np.sum(
        [singular_vals[i, 0] * tensor(u_mat[:, :, i], vt_mat[:, :, i]) for i in range(len(singular_vals))],
        axis=0,
    )
    assert np.isclose(rho, reconstructed).all()


def test_allclose_phi5():
    """Checks output of phi5 is close to expected."""
    phi5 = (
        (1 + np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_0)
        + (1 - np.sqrt(6)) / (2 * np.sqrt(6)) * np.kron(e_0, e_1)
        + (np.sqrt(2) - np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_0)
        + (np.sqrt(2) + np.sqrt(3)) / (2 * np.sqrt(6)) * np.kron(e_1, e_1)
    )
    calculated_singular_vals, calculated_u_mat, calculated_vt_mat = schmidt_decomposition(phi5)
    expected_singular_vals = np.array([[0.8660254], [0.5]])
    expected_u_mat = np.array([[-0.81649658, 0.57735027], [0.57735027, 0.81649658]])
    expected_v_mat = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
    np.testing.assert_allclose(calculated_singular_vals, expected_singular_vals, 1e-5)
    np.testing.assert_allclose(calculated_vt_mat, expected_v_mat, 1e-5)
    np.testing.assert_allclose(calculated_u_mat, expected_u_mat, 1e-5)
