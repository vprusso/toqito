"""Tests for is_unital."""
import numpy as np

from toqito.channel_props import is_unital
from toqito.channels import depolarizing
from toqito.perms import swap_operator


def test_is_unital_kraus_true():
    """Verify unital channel as Kraus ops as True."""
    kraus_1 = np.array([[1, 0], [0, 0]])
    kraus_2 = np.array([[1, 0], [0, 0]]).conj().T
    kraus_3 = np.array([[0, 1], [0, 0]])
    kraus_4 = np.array([[0, 1], [0, 0]]).conj().T
    kraus_5 = np.array([[0, 0], [1, 0]])
    kraus_6 = np.array([[0, 0], [1, 0]]).conj().T
    kraus_7 = np.array([[0, 0], [0, 1]])
    kraus_8 = np.array([[0, 0], [0, 1]]).conj().T

    kraus_ops = [
        [kraus_1, kraus_2],
        [kraus_3, kraus_4],
        [kraus_5, kraus_6],
        [kraus_7, kraus_8],
    ]
    np.testing.assert_equal(is_unital(kraus_ops), True)


def test_is_unital_swap_operator_choi_true():
    """Verify Choi matrix of the swap operator map is unital."""
    np.testing.assert_equal(is_unital(swap_operator(3)), True)


def test_is_unital_depolarizing_choi_false():
    """Verify Choi matrix of the depolarizing map is not unital."""
    np.testing.assert_equal(is_unital(depolarizing(4)), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
