import pytest
import numpy as np
from toqito.state_ops import renyi_entropy

# Example density matrices for testing
rho_pure = np.array([[1, 0], [0, 0]])  # Taking a pure state density matrix
rho_mixed = np.array([[0.5, 0], [0, 0.5]])  # Taking a maximum mixed state
invalid_rho = np.array([[0.5, 0], [0, 0.4]])  # Taking error case, trace not equals 1

def test_alpha_0():
    assert renyi_entropy(rho_pure, 0) == pytest.approx(0.0)
    assert renyi_entropy(rho_mixed, 0) == pytest.approx(np.log2(2))

def test_alpha_1():
    assert renyi_entropy(rho_pure, 1) == pytest.approx(0.0)
    assert renyi_entropy(rho_mixed, 1) == pytest.approx(1.0)

def test_alpha_inf():
    assert renyi_entropy(rho_pure, np.inf) == pytest.approx(0.0)
    assert renyi_entropy(rho_mixed, np.inf) == pytest.approx(1.0)

def test_invalid_density_matrix():
    with pytest.raises(ValueError, match="The density matrix must have trace equal to 1"):
        renyi_entropy(invalid_rho, 1)  # Should raise ValueError

def test_general_alpha():
    assert renyi_entropy(rho_mixed, 2) == pytest.approx(np.log2(0.5**2 + 0.5**2)/(1-2))
