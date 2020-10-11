"""Tests for kraus_to_choi."""
import numpy as np

from toqito.channels import dephasing, depolarizing
from toqito.channel_ops import kraus_to_choi


def test_kraus_to_choi_swap_operator():
    """Choi matrix of the transpose map is the swap operator."""
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

    choi_res = kraus_to_choi(kraus_ops)
    expected_choi_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    bool_mat = np.isclose(choi_res, expected_choi_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_kraus_to_choi_swap_operator_non_unique():
    """As Kraus operators are non-unique, these also should yield the swap operator."""
    kraus_1 = np.array([[0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]])
    kraus_2 = np.array([[0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]]).conj().T
    kraus_3 = np.array([[1, 0], [0, 0]])
    kraus_4 = np.array([[1, 0], [0, 0]]).conj().T
    kraus_5 = np.array([[0, 0], [0, 1]])
    kraus_6 = np.array([[0, 0], [0, 1]]).conj().T
    kraus_7 = np.array([[0, 1 / np.sqrt(2)], [-1 / np.sqrt(2), 0]])
    kraus_8 = np.array([[0, 1 / np.sqrt(2)], [-1 / np.sqrt(2), 0]]).conj().T

    kraus_ops = [
        [kraus_1, kraus_2],
        [kraus_3, kraus_4],
        [kraus_5, kraus_6],
        [kraus_7, kraus_8],
    ]

    choi_res = kraus_to_choi(kraus_ops)
    expected_choi_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    bool_mat = np.isclose(choi_res, expected_choi_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_kraus_to_choi_dephasing_channel():
    """Kraus operators for dephasing channel should yield the proper Choi matrix."""
    kraus_1 = np.array([[1, 0], [0, 0]])
    kraus_2 = np.array([[1, 0], [0, 0]])
    kraus_3 = np.array([[0, 0], [0, 1]])
    kraus_4 = np.array([[0, 0], [0, 1]])

    kraus_ops = [[kraus_1, kraus_2], [kraus_3, kraus_4]]

    choi_res = kraus_to_choi(kraus_ops)
    expected_choi_res = dephasing(2)

    bool_mat = np.isclose(choi_res, expected_choi_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_kraus_to_choi_depolarizing_channel():
    """Kraus operators for depolarizing channel should yield the proper Choi matrix."""
    kraus_1 = np.array([[1 / np.sqrt(2), 0], [0, 0]])
    kraus_2 = np.array([[1 / np.sqrt(2), 0], [0, 0]])
    kraus_3 = np.array([[0, 0], [1 / np.sqrt(2), 0]])
    kraus_4 = np.array([[0, 0], [1 / np.sqrt(2), 0]])
    kraus_5 = np.array([[0, 1 / np.sqrt(2)], [0, 0]])
    kraus_6 = np.array([[0, 1 / np.sqrt(2)], [0, 0]])
    kraus_7 = np.array([[0, 0], [0, 1 / np.sqrt(2)]])
    kraus_8 = np.array([[0, 0], [0, 1 / np.sqrt(2)]])

    kraus_ops = [
        [kraus_1, kraus_2],
        [kraus_3, kraus_4],
        [kraus_5, kraus_6],
        [kraus_7, kraus_8],
    ]

    choi_res = kraus_to_choi(kraus_ops)
    expected_choi_res = depolarizing(2)

    bool_mat = np.isclose(choi_res, expected_choi_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
