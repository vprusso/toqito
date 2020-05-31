"""Test cnot."""
import numpy as np

from toqito.matrices import cnot


def test_cnot():
    """Test standard CNOT gate."""
    res = cnot()
    expected_res = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
