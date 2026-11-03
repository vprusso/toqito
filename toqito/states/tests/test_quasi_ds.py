"""Test quasi_ds."""

from math import comb

import numpy as np
import pytest

from toqito.matrix_ops import partial_transpose
from toqito.matrix_props import is_density
from toqito.perms import permute_systems
from toqito.state_props import is_ppt
from toqito.states import dicke, quasi_ds


@pytest.mark.parametrize("num_qubits", [5, 7])
@pytest.mark.parametrize("z_param", [0.5, 3.0])
@pytest.mark.parametrize("sigma", [1, -1])
def test_quasi_ds_density_and_ppt_across_every_cut(num_qubits, z_param, sigma):
    """The state is a density operator and PPT across every bipartition of the qubits."""
    rho = quasi_ds(num_qubits, z_param, sigma)
    assert is_density(rho)
    for num_left in range(1, num_qubits // 2 + 1):
        assert is_ppt(rho, dim=[2**num_left, 2 ** (num_qubits - num_left)])


def test_quasi_ds_rank_pattern():
    """The ranks of the state and of its partial transposes across the k|n-k cuts are (n+1, 2n, ..., 2n, 2n-1)."""
    for num_qubits in (5, 7):
        rho = quasi_ds(num_qubits, 1.0, 1)
        ranks = [np.linalg.matrix_rank(rho)]
        for num_left in range(1, num_qubits // 2 + 1):
            dims = [2**num_left, 2 ** (num_qubits - num_left)]
            ranks.append(np.linalg.matrix_rank(partial_transpose(rho, [1], dims)))
        assert ranks == [num_qubits + 1] + [2 * num_qubits] * (num_qubits // 2 - 1) + [2 * num_qubits - 1]


def test_quasi_ds_permutation_symmetric():
    """The state is invariant under permuting the qubits."""
    rho = quasi_ds(5, 2.0, -1)
    np.testing.assert_allclose(permute_systems(rho, [1, 0, 2, 3, 4], [2] * 5), rho, atol=1e-12)
    np.testing.assert_allclose(permute_systems(rho, [4, 3, 2, 1, 0], [2] * 5), rho, atol=1e-12)


def test_quasi_ds_dicke_basis_entries():
    """In the Dicke basis the state has the diagonal C(n,k) f_{K-k}(Z) and corners sigma, over 2(4+Z)^K."""
    num_qubits, z_param, sigma = 5, 1.5, -1
    rho = quasi_ds(num_qubits, z_param, sigma)
    isometry = np.column_stack([dicke(num_qubits, k) for k in range(num_qubits + 1)])
    rho_dicke = isometry.T @ rho @ isometry
    # f_0 = 1, f_1 = 1 + Z, f_2 = (2 + Z) f_1 - f_0, and f_{-k} = f_{k-1}.
    f_vals = {0: 1.0, 1: 1.0 + z_param}
    f_vals[2] = (2 + z_param) * f_vals[1] - f_vals[0]
    f_vals[-1], f_vals[-2], f_vals[-3] = f_vals[0], f_vals[1], f_vals[2]
    norm = 2 * (4 + z_param) ** 2
    expected = np.diag([comb(num_qubits, k) * f_vals[2 - k] for k in range(num_qubits + 1)]) / norm
    expected[0, num_qubits] = expected[num_qubits, 0] = sigma / norm
    np.testing.assert_allclose(rho_dicke, expected, atol=1e-12)
    # The state is supported on the symmetric subspace, so this is the whole state.
    np.testing.assert_allclose(isometry @ rho_dicke @ isometry.T, rho, atol=1e-12)


@pytest.mark.parametrize(
    "num_qubits, z_param, sigma", [(4, 1.0, 1), (3, 1.0, 1), (5, 0.0, 1), (5, -1.0, 1), (5, 1.0, 2)]
)
def test_quasi_ds_invalid(num_qubits, z_param, sigma):
    """Even or too small qubit numbers, non-positive Z, and signs other than +-1 are rejected."""
    with pytest.raises(ValueError):
        quasi_ds(num_qubits, z_param, sigma)
