"""Test smolin."""

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.perms import permute_systems
from toqito.state_props import is_ppt, is_separable
from toqito.states import bell, generalized_smolin, smolin

QUBITS = [2, 2, 2, 2]


def test_smolin_is_bell_mixture():
    """The Smolin state is the equal mixture of |phi_i><phi_i| (x) |phi_i><phi_i| over the Bell states."""
    rho = smolin()
    mixture = sum(np.kron(bell(i) @ bell(i).conj().T, bell(i) @ bell(i).conj().T) for i in range(4)) / 4
    np.testing.assert_allclose(rho, mixture, atol=1e-12)
    assert is_density(rho)
    np.testing.assert_allclose(rho, generalized_smolin(4))


@pytest.mark.parametrize("perm", [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]])
def test_smolin_separable_across_two_two_cuts(perm):
    """Across each 2|2 cut (AB|CD, AC|BD, AD|BC) the state is PPT and certified separable."""
    rho = permute_systems(smolin(), perm, QUBITS)
    assert is_ppt(rho, dim=[4, 4])
    separable, witness = is_separable(rho, dim=[4, 4])
    assert separable
    assert "Horodecki" in witness


@pytest.mark.parametrize("perm", [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]])
def test_smolin_npt_across_one_three_cuts(perm):
    """Across each 1|3 cut the state has negative partial transpose."""
    rho = permute_systems(smolin(), perm, QUBITS)
    assert not is_ppt(rho, dim=[2, 8])


def test_smolin_partial_transpose_spectrum():
    """The 1|3 partial transpose has eigenvalue 1/8 with multiplicity 12 and -1/8 with multiplicity 4."""
    eigs = np.sort(np.linalg.eigvalsh(partial_transpose(smolin(), [1], [2, 8])))
    np.testing.assert_allclose(eigs, [-1 / 8] * 4 + [1 / 8] * 12, atol=1e-12)
