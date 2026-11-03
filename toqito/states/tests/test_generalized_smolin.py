"""Test generalized_smolin."""

import numpy as np
import pytest

from toqito.matrices import pauli
from toqito.matrix_props import is_density
from toqito.state_props import is_ppt, is_separable
from toqito.states import generalized_smolin


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_generalized_smolin_definition(num_qubits):
    """The state is (I + (-1)^n sum_P P^(x)2n) / 2^(2n) and a density operator."""
    sign = (-1) ** (num_qubits // 2)
    expected = np.identity(2**num_qubits) + sign * sum(pauli([i] * num_qubits) for i in (1, 2, 3))
    rho = generalized_smolin(num_qubits)
    np.testing.assert_allclose(rho, expected / 2**num_qubits, atol=1e-12)
    assert is_density(rho)


def test_generalized_smolin_two_qubits_is_singlet():
    """For two qubits the construction gives the singlet state."""
    singlet = np.array([[0], [1], [-1], [0]]) / np.sqrt(2)
    np.testing.assert_allclose(generalized_smolin(2), singlet @ singlet.T, atol=1e-12)


def test_generalized_smolin_six_qubit_cuts():
    """Six qubits: PPT and separable across the even 2|4 cut, NPT across the odd 1|5 and 3|3 cuts."""
    rho = generalized_smolin(6)
    assert is_ppt(rho, dim=[4, 16])
    separable, witness = is_separable(rho, dim=[4, 16])
    assert separable
    assert "Horodecki" in witness
    assert not is_ppt(rho, dim=[2, 32])
    assert not is_ppt(rho, dim=[8, 8])


def test_generalized_smolin_wrong_sign_is_not_positive():
    """With the opposite sign the matrix has negative eigenvalues, so the (-1)^n is forced."""
    num_qubits = 6
    wrong = np.identity(64) - (-1) ** 3 * sum(pauli([i] * num_qubits) for i in (1, 2, 3))
    assert np.linalg.eigvalsh(wrong).min() < 0


@pytest.mark.parametrize("num_qubits", [0, -2, 3])
def test_generalized_smolin_invalid(num_qubits):
    """Only positive even numbers of qubits are accepted."""
    with pytest.raises(ValueError):
        generalized_smolin(num_qubits)
