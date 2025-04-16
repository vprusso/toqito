"""Tests for operator_sinkhorn."""

import warnings

import numpy as np
import pytest

from toqito.channels.partial_trace import partial_trace
from toqito.matrices.operator_sinkhorn import operator_sinkhorn
from toqito.rand import random_density_matrix
from toqito.states import bell


def test_operator_sinkhorn_unitary_invariance():
    """Test invariance of Operator Sinkhorn on swapping subsystems."""
    rho = random_density_matrix(4)

    U = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))  # Swap subsystem
    rho_new = U @ rho @ U.conj().T
    sigma_new, F_new = operator_sinkhorn(rho_new)
    sigma_old, F_old = operator_sinkhorn(rho)
    np.testing.assert_allclose(sigma_new, U @ sigma_old @ U.conj().T)


def test_operator_sinkhorn_trace_property():
    """Test the trace property of operator Sinkhorn when a valid inputs are passed."""
    rho = random_density_matrix(9)  # 9-dimensional density matrix

    # The dimension `3` divides evenly into `9`, so no error should occur
    sigma, F = operator_sinkhorn(rho, dim=[3])

    # Check that sigma is a valid density matrix with trace equal to 1
    np.testing.assert_almost_equal(np.trace(sigma), 1)


# Test partial traces for bipartites and tripartites systems

"""Test operator Sinkhorn partial trace on a bipartite system."""
# Generate a random density matrix for a 3 qubit x 3qubit system (9-dimensional).
rho_bi = random_density_matrix(9)
dim_bi = [3, 3]
expected_identity_bi = np.eye(3) * (1 / 3)
subsystem_bi = 0  # first subsystem only

"""Test operator Sinkhorn partial trace on a tripartite system."""
rho_tri = random_density_matrix(8)
dim_tri = [2, 2, 2]
expected_identity_tri = np.eye(2) * (1 / 2)
subsystem_tri = [0, 2]

params0 = [
    ([rho_bi, dim_bi], [expected_identity_bi, subsystem_bi]),
    ([rho_tri, dim_tri], [expected_identity_tri, subsystem_tri]),
]


@pytest.mark.parametrize("test_input,expected", params0)
def test_operator_sinkhorn_partial_trace(test_input, expected):
    """Test partial traces for bipartites and tripartites systems."""
    rho1, dim1 = test_input
    sigma, F = operator_sinkhorn(rho=rho1, dim=dim1)

    # Expected partial trace should be proportional to identity matrix.
    expected_identity, target_subsys = expected

    # Partial trace on the first subsystem.
    pt = partial_trace(sigma, target_subsys, dim=dim1)
    pt_rounded = np.around(pt, decimals=2)

    # Check that partial trace matches the expected identity.
    np.testing.assert_allclose(pt_rounded, expected_identity, rtol=1e-2)


# Test Error handling in different scenarios

"""Test operator Sinkhorn with a singular matrix that triggers LinAlgError."""
rho_singular = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])  # This matrix is singular
dim_singular = [2, 2]
expected_msg_singular = (
    "The operator Sinkhorn iteration does not converge for rho. This is often the case if rho is not of full rank."
)

"""Test operator Sinkhorn when a single number is passed as `dim` and it is invalid."""
rho_invalid_sdim = random_density_matrix(8)  # 8-dimensional density matrix
# The dimension `3` does not divide evenly into `8`, so we expect an error
dim_invalid_sdim = [3]
expected_msg_invalid_sdim = (
    "If dim is of size 1, rho must be square and dim[0] must evenly divide rho.shape[0]; "
    "please provide the dim array containing the dimensions of the subsystems."
)

"""Test operator Sinkhorn when product of the dim array does not match the density matrix dims."""
rho_invalid_dimarr = random_density_matrix(8)
# 4*3*2 != 8. So dim array should not match
dim_invalid_dimarr = [4, 3, 2]
expected_msg_invalid_dimarr = " ".join(
    (f"Product of dimensions {dim_invalid_dimarr}", f"does not match rho dimension {rho_invalid_dimarr.shape[0]}.")
)

"""Test operator sinkhorn on non-square input matrix."""
rho_nonsquare = np.random.rand(4, 5)
dim_nonsquare = [2, 2]
expected_msg_nonsquare = "Input 'rho' must be a square matrix."


params1 = [
    ([rho_singular, dim_singular], expected_msg_singular),  # singular matrix
    ([rho_invalid_sdim, dim_invalid_sdim], expected_msg_invalid_sdim),  # invalid single dim
    ([rho_invalid_dimarr, dim_invalid_dimarr], expected_msg_invalid_dimarr),  # invalid dim array
    ([rho_nonsquare, dim_nonsquare], expected_msg_nonsquare),  # non square matrix input
]


@pytest.mark.parametrize("test_input,expected_msg", params1)
def test_operator_sinkhorn_errors(test_input, expected_msg):
    """Test Error handling in different scenarios."""
    # unpack test inputs
    rho, dim1 = test_input

    try:
        operator_sinkhorn(rho, dim=dim1)
    except ValueError as e:
        assert str(e) == expected_msg


def test_operator_sinkhorn_max_iterations():
    """Test operator sinkhorn on insufficient iteration limit."""

    rho_random = random_density_matrix(4, seed=42)
    try:
        operator_sinkhorn(rho=rho_random, dim=[2, 2], max_iterations=20)
    except RuntimeError as e:
        expected_msg = "operator_sinkhorn did not converge within 20 iterations."
        assert str(e) == expected_msg


def test_operator_sinkhorn_near_zero_trace():
    """Test if operator sinkhorn raises warnings on near zero output trace."""
    rho_random = np.array(
        [
            [np.finfo(float).eps, 0, 0, 0],
            [0, np.finfo(float).eps, 0, 0],
            [0, 0, np.finfo(float).eps, 0],
            [0, 0, 0, np.finfo(float).eps],
        ]
    )
    expected_msg = "Final trace is near zero, but initial trace was not. Result may be unreliable."

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            operator_sinkhorn(rho=rho_random, dim=[2, 2])
        except RuntimeWarning as warn_string:
            assert str(warn_string) == expected_msg


# Test outputs on maximally mixed and entangled states

"""Test operator Sinkhorn on a maximally mixed bipartite state. Result should be invariant."""
rho_mixed = np.eye(9)  # Identity matrix is a max-mixed state
# The dimension `3` divides evenly into `9`, so no error should occur
dim_mixed = [3]

"""Test operator Sinkhorn on a maximally entangled bipartite state. Should be invariant."""
# function should return the inintial state since it already satisfies the trace property
u0 = bell(0)
rho_ent = u0 @ u0.conj().T
dim_ent = [2]

params2 = [([rho_mixed, dim_mixed]), ([rho_ent, dim_ent])]


@pytest.mark.parametrize("test_input", params2)
def test_operator_sinkhorn_mix_ent(test_input):
    """Test outputs on maximally mixed and entangled states."""
    rho1, dim1 = test_input

    sigma, F = operator_sinkhorn(rho=rho1, dim=dim1)

    # Check that rho is invariant after sinkhorn operation
    np.testing.assert_allclose(sigma, rho1)
