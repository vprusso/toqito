"""Test ppt_distinguishability."""
import numpy as np
import pytest

from toqito.perms import swap_operator
from toqito.state_opt import ppt_distinguishability
from toqito.states import basis, bell

psi_0_YYD = bell(0)
psi_1_YYD = bell(2)
psi_2_YYD = bell(3)
psi_3_YYD = bell(1)

x_1_YYD = np.kron(psi_0_YYD, psi_0_YYD)
x_2_YYD = np.kron(psi_1_YYD, psi_3_YYD)
x_3_YYD = np.kron(psi_2_YYD, psi_3_YYD)
x_4_YYD = np.kron(psi_3_YYD, psi_3_YYD)

rho_1_YYD = x_1_YYD * x_1_YYD.conj().T
rho_2_YYD = x_2_YYD * x_2_YYD.conj().T
rho_3_YYD = x_3_YYD * x_3_YYD.conj().T
rho_4_YYD = x_4_YYD * x_4_YYD.conj().T

states_YYD = [rho_1_YYD, rho_2_YYD, rho_3_YYD, rho_4_YYD]
vec_YYD = [x_1_YYD, x_2_YYD, x_3_YYD, x_4_YYD]
probs_YYD = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

# Werner hiding pairs
dim = 2
sigma_0 = (np.kron(np.identity(dim), np.identity(dim)) + swap_operator(dim)) / (dim * (dim + 1))
sigma_1 = (np.kron(np.identity(dim), np.identity(dim)) - swap_operator(dim)) / (dim * (dim - 1))
states_werner = [sigma_0, sigma_1]
expected_val_werner = 1 / 2 + 1 / (dim + 1)

# 4 bell states with a resource state
rho_1 = bell(0) * bell(0).conj().T
rho_2 = bell(1) * bell(1).conj().T
rho_3 = bell(2) * bell(2).conj().T
rho_4 = bell(3) * bell(3).conj().T
e_0, e_1 = basis(2, 0), basis(2, 1)
e_00 = np.kron(e_0, e_0)
e_11 = np.kron(e_1, e_1)

eps = 0.5
resource_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
resource_state = resource_state * resource_state.conj().T

states_bell_resource = [
    np.kron(rho_1, resource_state),
    np.kron(rho_2, resource_state),
    np.kron(rho_3, resource_state),
    np.kron(rho_4, resource_state),
]
probs_bell = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

exp_res_bell = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

@pytest.mark.parametrize("input_states, input_probs, dist_method_str, strategy_bool, expected_result", [
    # PPT distinguishing the YYD states from :cite:`Yu_2012_Four` should yield `7/8 ~ 0.875`.
    # Feeding the input to the function as density matrices.
    # Primal result - min error
    (states_YYD, probs_YYD, "min-error", True, 7/8),
    # Dual Result - min error
    (states_YYD, probs_YYD, "min-error", False, 7/8),
    # Primal result - unambiguous
    (states_YYD, probs_YYD, "unambiguous", True, 3/4),
    # Dual Result - unambiguous
    (states_YYD, probs_YYD, "unambiguous", False, 3/4),
    # PPT distinguishing the YYD states from :cite:`Yu_2012_Four` should yield `7/8 ~ 0.875`.
    # Feeding the input to the function as vectors.
    # Primal result - min error
    (vec_YYD, probs_YYD, "min-error", True, 7/8),
    # Dual Result - min error
    (vec_YYD, probs_YYD, "min-error", False, 7/8),
    # Primal result - unambiguous
    (vec_YYD, probs_YYD, "unambiguous", True, 3/4),
    # Dual Result - unambiguous
    (vec_YYD, probs_YYD, "unambiguous", False, 3/4),
    # PPT distinguishing the YYD states from :cite:`Yu_2012_Four` should yield `7/8 ~ 0.875`.
    # Feeding the input to the function as density matrices.
    # Primal result - min error without probs
    (states_YYD, None, "min-error", True, 7/8),
    # Dual Result - min error without probs
    (states_YYD, None, "min-error", False, 7/8),
    # Primal result - unambiguous without probs
    (states_YYD, None, "unambiguous", True, 3/4),
    # Dual Result - unambiguous without probs
    (states_YYD, None, "unambiguous", False, 3/4),
    # One quantum data hiding scheme involves the Werner hiding pair :cite:`Terhal_2001_Hiding`. A Werner hiding pair
    # is defined by :cite:`Cosentino_2015_QuantumState`.
    # Primal result - min error without probs
    (states_werner, None, "min-error", True, expected_val_werner),
    # Dual Result - min error without probs
    (states_werner, None, "min-error", False, expected_val_werner),
    # Primal result - unambiguous without probs
    (states_werner, None, "unambiguous", True, 1/3),
    # Dual Result - unambiguous without probs
    (states_werner, None, "unambiguous", False, 1/3),
    # PPT distinguishing the four Bell states. There exists a closed form formula for the probability with which one
    # is able to distinguish one of the four Bell states given with equal
    # probability when Alice and Bob have access to a resource state :cite:`Bandyopadhyay_2015_Limitations`.
    # Refer to Theorem 5 in  :cite:`Bandyopadhyay_2015_Limitations` for more details.
    # Primal result - min error
    (states_bell_resource, probs_bell, "min-error", True, exp_res_bell),
    # Dual result - min error
    (states_bell_resource, probs_bell, "min-error", False, exp_res_bell)])
def test_ppt_distinguishability(input_states, input_probs, dist_method_str, strategy_bool, expected_result):
    """Tests function works as expected for valid inputs."""
    calculated_result = ppt_distinguishability(
        input_states, probs=input_probs, dist_method=dist_method_str, strategy=strategy_bool)
    assert np.isclose(calculated_result, expected_result, atol = 0.001)

