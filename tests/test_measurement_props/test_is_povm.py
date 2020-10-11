"""Test is_povm."""
import numpy as np

from toqito.measurement_props import is_povm
from toqito.random import random_povm


def test_is_povm_true():
    """Test if valid measurement returns True."""
    dim, num_inputs, num_outputs = 2, 2, 2
    measurements = random_povm(dim, num_inputs, num_outputs)

    np.testing.assert_equal(is_povm([measurements[:, :, 0, 0], measurements[:, :, 0, 1]]), True)


def test_is_povm_false():
    """Test if invalid POVM returns False."""
    non_meas_1 = np.array([[1, 2], [3, 4]])
    non_meas_2 = np.array([[5, 6], [7, 8]])
    non_meas = [non_meas_1, non_meas_2]

    np.testing.assert_equal(is_povm(non_meas), False)


def test_is_povm_false_not_sum_identity():
    """Test if invalid POVM (does not sum to identity)."""
    non_meas_1 = np.array([[1, 0], [0, 1]])
    non_meas_2 = np.array([[1, 0], [0, 1]])
    non_meas = [non_meas_1, non_meas_2]

    np.testing.assert_equal(is_povm(non_meas), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
