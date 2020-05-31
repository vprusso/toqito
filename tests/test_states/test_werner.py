"""Test werner."""
import numpy as np

from toqito.states import werner


def test_werner_qutrit():
    """Test for qutrit Werner state."""
    res = werner(3, 1 / 2)
    np.testing.assert_equal(np.isclose(res[0][0], 0.0666666), True)
    np.testing.assert_equal(np.isclose(res[1][3], -0.066666), True)


def test_werner_multipartite():
    """Test for multipartite Werner state."""
    res = werner(2, [0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_equal(np.isclose(res[0][0], 0.1127, atol=1e-02), True)


def test_werner_invalid_alpha():
    """Test for invalid `alpha` parameter."""
    with np.testing.assert_raises(ValueError):
        werner(3, [1, 2])


if __name__ == "__main__":
    np.testing.run_module_suite()
