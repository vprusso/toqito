"""Test ppt_distinguishability."""

import numpy as np
import pytest

from toqito.state_opt import ppt_distinguishability
from toqito.states import basis, bell


def test_ppt_distinguishability_yyd_density_matrices():
    """PPT distinguishing the YYD states from :footcite:`Yu_2012_Four` should yield `7/8 ~ 0.875`.

    Feeding the input to the function as density matrices.
    """
    psi_0 = bell(0)
    psi_1 = bell(2)
    psi_2 = bell(3)
    psi_3 = bell(1)

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    rho_1 = x_1 @ x_1.conj().T
    rho_2 = x_2 @ x_2.conj().T
    rho_3 = x_3 @ x_3.conj().T
    rho_4 = x_4 @ x_4.conj().T

    states = [rho_1, rho_2, rho_3, rho_4]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    # Min-error tests:
    primal_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="min_error",
        primal_dual="primal",
    )
    dual_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="min_error",
        primal_dual="dual",
    )
    assert np.isclose(primal_res, 7 / 8, atol=0.001)
    assert np.isclose(dual_res, 7 / 8, atol=0.001)

    primal_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="unambig",
        primal_dual="primal",
    )

    assert np.isclose(primal_res, 3 / 4, atol=0.001)


def test_ppt_distinguishability_yyd_vectors():
    """PPT distinguishing the YYD states from :footcite:`Yu_2012_Four` should yield `7/8 ~ 0.875`.

    Feeding the input to the function as state vectors.
    """
    psi_0 = bell(0)
    psi_1 = bell(2)
    psi_2 = bell(3)
    psi_3 = bell(1)

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    states = [x_1, x_2, x_3, x_4]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    # Min-error tests:
    primal_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="min_error",
        primal_dual="primal",
    )
    dual_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="min_error",
        primal_dual="dual",
    )

    assert np.isclose(primal_res, 7 / 8, atol=0.001)
    assert np.isclose(dual_res, 7 / 8, atol=0.001)

    primal_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="unambig",
        primal_dual="primal",
    )

    assert np.isclose(primal_res, 3 / 4, atol=0.001)


def test_ppt_distinguishability_yyd_states_no_probs():
    """PPT distinguishing the YYD states from :footcite:`Yu_2012_Four` should yield `7/8 ~ 0.875`.

    If no probability vector is explicitly given, assume uniform probabilities are given.
    """
    psi_0 = bell(0)
    psi_1 = bell(2)
    psi_2 = bell(3)
    psi_3 = bell(1)

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    rho_1 = x_1 @ x_1.conj().T
    rho_2 = x_2 @ x_2.conj().T
    rho_3 = x_3 @ x_3.conj().T
    rho_4 = x_4 @ x_4.conj().T

    states = [rho_1, rho_2, rho_3, rho_4]

    # Min-error tests:
    primal_res, _ = ppt_distinguishability(
        vectors=states, subsystems=[0, 2], dimensions=[2, 2, 2, 2], strategy="min_error", primal_dual="primal"
    )
    dual_res, _ = ppt_distinguishability(
        vectors=states, subsystems=[0, 2], dimensions=[2, 2, 2, 2], strategy="min_error", primal_dual="dual"
    )

    assert np.isclose(primal_res, 7 / 8, atol=0.001)
    assert np.isclose(dual_res, 7 / 8, atol=0.001)

    primal_res, _ = ppt_distinguishability(
        vectors=states, subsystems=[0, 2], dimensions=[2, 2, 2, 2], strategy="unambig", primal_dual="primal"
    )

    assert np.isclose(primal_res, 3 / 4, atol=0.001)


def test_ppt_distinguishability_four_bell_states():
    r"""PPT distinguishing the four Bell states.

    There exists a closed form formula for the probability with which one
    is able to distinguish one of the four Bell states given with equal
    probability when Alice and Bob have access to a resource state :footcite:`Bandyopadhyay_2015_Limitations`.

    The resource state is defined by

    .. math::
        |\tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} +
        |0\rangle | 0\rangle +
        \sqrt{\frac{1-\epsilon}{2}} |1 \rangle |1 \rangle

    The closed form probability with which Alice and Bob can distinguish via
    PPT measurements is given as follows

    .. math::
        \frac{1}{2} \left(1 + \sqrt{1 - \epsilon^2} \right).

    This formula happens to be equal to LOCC and SEP as well for this case.
    Refer to Theorem 5 in  :footcite:`Bandyopadhyay_2015_Limitations` for more details.
    """
    rho_1 = bell(0)
    rho_2 = bell(1)
    rho_3 = bell(2)
    rho_4 = bell(3)

    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    eps = 0.5
    resource_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11

    states = [
        np.kron(rho_1, resource_state),
        np.kron(rho_2, resource_state),
        np.kron(rho_3, resource_state),
        np.kron(rho_4, resource_state),
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps**2))

    # Min-error tests:
    primal_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="min_error",
        primal_dual="primal",
    )
    dual_res, _ = ppt_distinguishability(
        vectors=states,
        probs=probs,
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        strategy="min_error",
        primal_dual="dual",
    )
    assert np.isclose(primal_res, exp_res, atol=0.001)
    assert np.isclose(dual_res, exp_res, atol=0.001)


@pytest.mark.parametrize(
    "vectors, probs, solver, subsystems, dimensions, strategy, primal_dual",
    [
        # Bell states (default uniform probs with dual).
        ([bell(0), bell(1), bell(2), np.array([[1], [0]])], None, "cvxopt", [0], [2, 2], "min_error", "dual"),
    ],
)
def test_ppt_state_distinguishability_invalid_vectors(
    vectors, probs, solver, subsystems, dimensions, strategy, primal_dual
):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError, match="Vectors for state distinguishability must all have the same dimension."):
        ppt_distinguishability(
            vectors=vectors,
            probs=probs,
            subsystems=subsystems,
            dimensions=dimensions,
            strategy=strategy,
            solver=solver,
            primal_dual=primal_dual,
        )


@pytest.mark.parametrize(
    "vectors, probs, solver, subsystems, dimensions, strategy, primal_dual",
    [
        # Bell states (default uniform probs with dual).
        ([bell(0), bell(1), bell(2), bell(3)], None, "cvxopt", [0], [2, 2], "unambig", "dual"),
    ],
)
def test_ppr_state_distinguishability_invalid_strategy(
    vectors, probs, solver, subsystems, dimensions, strategy, primal_dual
):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError, match="Minimum-error PPT distinguishability only supported at this time."):
        ppt_distinguishability(
            vectors=vectors,
            probs=probs,
            subsystems=subsystems,
            dimensions=dimensions,
            strategy=strategy,
            solver=solver,
            primal_dual=primal_dual,
        )
