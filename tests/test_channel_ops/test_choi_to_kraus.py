"""Tests for choi_to_kraus."""
import numpy as np

from toqito.channel_ops import choi_to_kraus


def test_choi_to_kraus():
    """Choi matrix of the swap operator."""

    choi_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    kraus_ops = [
        np.array([[0, 1j / np.sqrt(2)], [-1j / np.sqrt(2), 0]]),
        np.array([[0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
    ]
    res_kraus_ops = choi_to_kraus(choi_mat)

    bool_mat = np.isclose(kraus_ops[0], res_kraus_ops[0])
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(kraus_ops[1], res_kraus_ops[1])
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(kraus_ops[2], res_kraus_ops[2])
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(kraus_ops[3], res_kraus_ops[3])
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
