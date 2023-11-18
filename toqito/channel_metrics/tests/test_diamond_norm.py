"""Tests for diamond_norm."""
import numpy as np

from toqito.channel_metrics import diamond_norm
from toqito.channels import dephasing, depolarizing


def test_diamond_norm_same_channel():
    """The diamond norm of identical channels should yield 0."""
    choi_1 = dephasing(2)
    choi_2 = dephasing(2)
    np.testing.assert_equal(np.isclose(diamond_norm(choi_1, choi_2), 0, atol=1e-3), True)


def test_diamond_norm_different_channel():
    """Calculate the diamond norm of different channels."""
    choi_1 = dephasing(2)
    choi_2 = depolarizing(2)
    np.testing.assert_equal(np.isclose(diamond_norm(choi_1, choi_2), 1, atol=1e-3), True)


def test_diamond_norm_inconsistent_dims():
    """Inconsistent dimensions between Choi matrices."""
    with np.testing.assert_raises(ValueError):
        choi_1 = depolarizing(4)
        choi_2 = dephasing(2)
        diamond_norm(choi_1, choi_2)


def test_diamond_norm_non_square():
    """Non-square inputs for diamond norm."""
    with np.testing.assert_raises(ValueError):
        choi_1 = np.array([[1, 2, 3], [4, 5, 6]])
        choi_2 = np.array([[1, 2, 3], [4, 5, 6]])
        diamond_norm(choi_1, choi_2)


if __name__ == "__main__":
    np.testing.run_module_suite()
