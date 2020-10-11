"""Test breuer."""
import numpy as np

from toqito.states import breuer


def test_breuer_dim_4_ent():
    """Generate Breuer state of dimension 4 with weight 0.1."""
    expected_res = np.array([[0.3, 0, 0, 0], [0, 0.2, 0.1, 0], [0, 0.1, 0.2, 0], [0, 0, 0, 0.3]])

    res = breuer(2, 0.1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_breuer_invalid_dim():
    """Ensures that an odd dimension if not accepted."""
    with np.testing.assert_raises(ValueError):
        breuer(3, 0.1)


if __name__ == "__main__":
    np.testing.run_module_suite()
