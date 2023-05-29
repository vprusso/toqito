import numpy as np
from toqito.channels import dephasing
def test_diamond_norm_unitary():
    """The diamond norm of a quantum channel is 1"""
    phi = dephasing(1)
    np.testing.assert_equal(np.isclose(diamond_norm(phi), 1, atol=1e-3), True)
def test_diamond_norm_unitary():
    """The diamond norm of phi = id- U id U* is the diameter of the smallest circle that contains the eigenvalues of U"""
    U = 1 / np.sqrt(2) * np.array([[1, 1], [-1, 1]]) # Hadamard gate
    phi = np.array([[np.eye(2), np.eye(2)], [U, -U]])
    phi = phi.reshape((4, 4))
    lam, eigv = np.linalg.eig(U)
    diameter = np.abs(lam[0] - lam[1])
    np.testing.assert_equal(np.isclose(diamond_norm(phi), diameter, atol=1e-3), True)
def test_diamond_norm_non_square():
    """Non-square inputs for diamond norm."""
    with np.testing.assert_raises(ValueError):
        phi = np.array([[1, 2, 3], [4, 5, 6]])
        diamond_norm(phi)

if __name__ == "__main__":
    np.testing.run_module_suite()