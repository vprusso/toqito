"""Test pure_to_mixed."""
import numpy as np

from toqito.state_ops import pure_to_mixed
from toqito.states import bell


def test_pure_to_mixed_state_vector():
    """Convert pure state to mixed state vector."""
    expected_res = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )

    phi = bell(0)
    res = pure_to_mixed(phi)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pure_to_mixed_density_matrix():
    """Convert pure state to mixed state density matrix."""
    expected_res = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )

    phi = bell(0) * bell(0).conj().T
    res = pure_to_mixed(phi)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_invalid_pure_to_mixed_input():
    """Invalid arguments for pure_to_mixed."""
    with np.testing.assert_raises(ValueError):
        non_valid_input = np.array([[1 / 2, 0, 0, 1 / 2], [1 / 2, 0, 0, 1 / 2]])
        pure_to_mixed(non_valid_input)


if __name__ == "__main__":
    np.testing.run_module_suite()
