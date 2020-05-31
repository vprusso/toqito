"""Test is_ensemble."""
import numpy as np

from toqito.state_props import is_ensemble


def test_is_ensemble_true():
    """Test if valid ensemble returns True."""
    rho_0 = np.array([[0.5, 0], [0, 0]])
    rho_1 = np.array([[0, 0], [0, 0.5]])
    states = [rho_0, rho_1]
    np.testing.assert_equal(is_ensemble(states), True)


def test_is_non_ensemble_non_psd():
    """Test if non-valid ensemble returns False."""
    rho_0 = np.array([[0.5, 0], [0, 0]])
    rho_1 = np.array([[-1, -1], [-1, -1]])
    states = [rho_0, rho_1]
    np.testing.assert_equal(is_ensemble(states), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
