"""Tests for sandwiched_renyi_conditional_entropy."""

import numpy as np
import pytest

from toqito.state_props import (
    renyi_entropy,
    sandwiched_renyi_conditional_entropy,
    von_neumann_entropy,
)
from toqito.state_props.sandwiched_renyi_conditional_entropy import (
    _sandwiched_renyi_conditional_entropy_downarrow,
)
from toqito.states import bell, max_mixed

RHO_A = np.array([[0.8, 0.0], [0.0, 0.2]])
RHO_B = np.array([[0.3, 0.0], [0.0, 0.7]])
PRODUCT_STATE = np.kron(RHO_A, RHO_B)
MAX_ENTANGLED_STATE = bell(0) @ bell(0).conj().T


@pytest.mark.parametrize("variant", ["downarrow", "uparrow"])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
def test_sandwiched_renyi_conditional_entropy_product_state(variant: str, alpha: float):
    """For a product state, conditioning on B should recover the entropy of A."""
    expected = renyi_entropy(RHO_A, alpha) if alpha != 1 else von_neumann_entropy(RHO_A)
    result = sandwiched_renyi_conditional_entropy(PRODUCT_STATE, alpha, dim=[2, 2], variant=variant)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("variant", ["downarrow", "uparrow"])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
def test_sandwiched_renyi_conditional_entropy_maximally_entangled(variant: str, alpha: float):
    """For a maximally entangled two-qubit pure state, the entropy should be -1."""
    result = sandwiched_renyi_conditional_entropy(MAX_ENTANGLED_STATE, alpha, variant=variant)
    np.testing.assert_allclose(result, -1.0, atol=1e-6)


def test_sandwiched_renyi_conditional_entropy_scalar_dim():
    """Allow scalar subsystem dimensions for bipartite states."""
    result = sandwiched_renyi_conditional_entropy(np.kron(max_mixed(2, False), max_mixed(3, False)), 2.0, dim=2)
    np.testing.assert_allclose(result, 1.0, atol=1e-8)


@pytest.mark.parametrize("alpha", [0.7, 1.5, 2.0, 3.0])
def test_sandwiched_renyi_conditional_entropy_uparrow_dominates_downarrow(alpha: float):
    """The uparrow variant is a supremum, so it is at least the downarrow variant."""
    np.random.seed(42)
    mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    rho = mat @ mat.conj().T
    rho = rho / np.trace(rho)

    down = sandwiched_renyi_conditional_entropy(rho, alpha, dim=[2, 2], variant="downarrow")
    up = sandwiched_renyi_conditional_entropy(rho, alpha, dim=[2, 2], variant="uparrow")
    assert up >= down - 1e-6


def test_sandwiched_renyi_conditional_entropy_uparrow_classical_quantum():
    """On a classical-quantum state the uparrow variant matches a reference calculation.

    For rho_AB = sum_x p(x) |x><x|_A ⊗ rho_B^x with orthogonal |x>,
    the optimal sigma_B equals the conditional Rényi mixture
        sigma_B* ∝ (sum_x p(x)^alpha (rho_B^x)^alpha)^{1/alpha},
    and the uparrow value has a closed form that we cross-check numerically.
    """
    p = np.array([0.6, 0.4])
    rho_b_1 = np.diag([0.9, 0.1])
    rho_b_2 = np.diag([0.2, 0.8])
    rho = p[0] * np.kron(np.diag([1, 0]), rho_b_1) + p[1] * np.kron(np.diag([0, 1]), rho_b_2)
    alpha = 2.0

    up = sandwiched_renyi_conditional_entropy(rho, alpha, dim=[2, 2], variant="uparrow")

    # Closed form for classical-quantum states: H^↑_α(A|B) = α/(1-α) · log2 Tr[(sum_x p_x^α (rho_B^x)^α)^{1/α}]
    mixture = p[0] ** alpha * np.linalg.matrix_power(rho_b_1, int(alpha)) + p[1] ** alpha * np.linalg.matrix_power(
        rho_b_2, int(alpha)
    )
    expected = alpha / (1 - alpha) * np.log2(np.trace(_fractional_power(mixture, 1 / alpha)))
    np.testing.assert_allclose(up, expected, atol=1e-6)


def _fractional_power(mat: np.ndarray, power: float) -> np.ndarray:
    """Apply a fractional power to a PSD matrix."""
    eigs, vecs = np.linalg.eigh((mat + mat.conj().T) / 2)
    eigs = np.maximum(eigs, 0.0)
    return vecs @ np.diag(eigs**power) @ vecs.conj().T


@pytest.mark.parametrize("variant", ["downarrow", "uparrow"])
def test_sandwiched_renyi_conditional_entropy_alpha_one_is_conditional_vn(variant: str):
    """At alpha=1 both variants reduce to the conditional von Neumann entropy."""
    expected = von_neumann_entropy(PRODUCT_STATE) - von_neumann_entropy(RHO_B)
    result = sandwiched_renyi_conditional_entropy(PRODUCT_STATE, 1.0, dim=[2, 2], variant=variant)
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_sandwiched_renyi_conditional_entropy_uparrow_rejects_small_alpha():
    """The uparrow variant requires alpha >= 1/2 for a convex formulation."""
    with pytest.raises(ValueError, match="alpha >= 1/2"):
        sandwiched_renyi_conditional_entropy(PRODUCT_STATE, 0.25, dim=[2, 2], variant="uparrow")


def test_sandwiched_renyi_conditional_entropy_rejects_unknown_variant():
    """An unknown variant should raise a clear ValueError."""
    with pytest.raises(ValueError, match="downarrow"):
        sandwiched_renyi_conditional_entropy(PRODUCT_STATE, 1.5, dim=[2, 2], variant="sideways")


@pytest.mark.parametrize(
    "rho, alpha, dim, expected_msg",
    [
        (np.array([[1, 2], [3, 4]]), 1.5, [2, 1], "density operators"),
        (PRODUCT_STATE, 0.0, [2, 2], "positive orders"),
        (PRODUCT_STATE, 1.5, [4], "exactly two subsystem dimensions"),
        (PRODUCT_STATE, 1.5, [3, 2], "product of `dim`"),
        (max_mixed(6, False), 1.5, None, "Cannot infer bipartite subsystem dimensions"),
        (PRODUCT_STATE, 1.5, 3, "positive divisor"),
        (PRODUCT_STATE, 1.5, [2.5, 1.6], "integer subsystem dimensions"),
        (PRODUCT_STATE, 1.5, [-1, -4], "must be positive"),
    ],
)
def test_sandwiched_renyi_conditional_entropy_invalid_input(
    rho: np.ndarray, alpha: float, dim: int | list[int] | None, expected_msg: str
):
    """Invalid inputs should raise a clear ValueError."""
    with pytest.raises(ValueError, match=expected_msg):
        sandwiched_renyi_conditional_entropy(rho, alpha, dim=dim)


def test_sandwiched_downarrow_returns_minus_inf_on_disjoint_support_for_small_alpha():
    """Defensive support guard: if rho and I ⊗ rho_b have disjoint support, downarrow returns -inf for alpha < 1."""
    # Synthetic mismatched (rho, rho_b): rho lives on |0,1> while rho_b projects onto |0>,
    # making I ⊗ rho_b support |0,0>, |1,0> -- disjoint from rho's support.
    rho = np.kron(np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
    rho_b = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert _sandwiched_renyi_conditional_entropy_downarrow(rho, rho_b, alpha=0.5, dim_a=2) == float("-inf")


def test_sandwiched_downarrow_returns_minus_inf_when_support_leaks_for_large_alpha():
    """Defensive support guard: if supp(rho) is not in supp(I ⊗ rho_b), downarrow returns -inf for alpha > 1."""
    rho = np.kron(np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
    rho_b = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert _sandwiched_renyi_conditional_entropy_downarrow(rho, rho_b, alpha=2.0, dim_a=2) == float("-inf")
