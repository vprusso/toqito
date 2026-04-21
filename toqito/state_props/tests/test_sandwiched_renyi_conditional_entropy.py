"""Tests for sandwiched_renyi_conditional_entropy."""

import numpy as np
import pytest

from toqito.state_props import (
    renyi_entropy,
    sandwiched_renyi_conditional_entropy,
    von_neumann_entropy,
)
from toqito.states import bell, max_mixed

RHO_A = np.array([[0.8, 0.0], [0.0, 0.2]])
RHO_B = np.array([[0.3, 0.0], [0.0, 0.7]])
PRODUCT_STATE = np.kron(RHO_A, RHO_B)
MAX_ENTANGLED_STATE = bell(0) @ bell(0).conj().T


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
def test_sandwiched_renyi_conditional_entropy_product_state(alpha: float):
    """For a product state, conditioning on B should recover the entropy of A."""
    expected = renyi_entropy(RHO_A, alpha) if alpha != 1 else von_neumann_entropy(RHO_A)
    result = sandwiched_renyi_conditional_entropy(PRODUCT_STATE, alpha, dim=[2, 2])
    np.testing.assert_allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
def test_sandwiched_renyi_conditional_entropy_maximally_entangled(alpha: float):
    """For a maximally entangled two-qubit pure state, the entropy should be -1."""
    result = sandwiched_renyi_conditional_entropy(MAX_ENTANGLED_STATE, alpha)
    np.testing.assert_allclose(result, -1.0, atol=1e-8)


def test_sandwiched_renyi_conditional_entropy_scalar_dim():
    """Allow scalar subsystem dimensions for bipartite states."""
    result = sandwiched_renyi_conditional_entropy(np.kron(max_mixed(2, False), max_mixed(3, False)), 2.0, dim=2)
    np.testing.assert_allclose(result, 1.0, atol=1e-8)


def test_sandwiched_renyi_conditional_entropy_unsupported_uparrow():
    """The uparrow variant should fail clearly until the optimization is implemented."""
    with pytest.raises(ValueError, match="downarrow"):
        sandwiched_renyi_conditional_entropy(PRODUCT_STATE, 1.5, dim=[2, 2], variant="uparrow")


@pytest.mark.parametrize(
    "rho, alpha, dim, expected_msg",
    [
        (np.array([[1, 2], [3, 4]]), 1.5, [2, 1], "density operators"),
        (PRODUCT_STATE, 0.0, [2, 2], "positive orders"),
        (PRODUCT_STATE, 1.5, [4], "exactly two subsystem dimensions"),
        (PRODUCT_STATE, 1.5, [3, 2], "product of `dim`"),
        (max_mixed(6, False), 1.5, None, "Cannot infer bipartite subsystem dimensions"),
    ],
)
def test_sandwiched_renyi_conditional_entropy_invalid_input(
    rho: np.ndarray, alpha: float, dim: int | list[int] | None, expected_msg: str
):
    """Invalid inputs should raise a clear ValueError."""
    with pytest.raises(ValueError, match=expected_msg):
        sandwiched_renyi_conditional_entropy(rho, alpha, dim=dim)
