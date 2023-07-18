"""Test state_exclusion."""
import pytest
import numpy as np

from toqito.matrices import standard_basis
from toqito.states import bell
from toqito.state_opt import state_exclusion


def test_conclusive_state_exclusion_one_state_vec():
    """Conclusive state exclusion for single vector state."""
    e_0, e_1 = standard_basis(2)
    states = [
        1/np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1)),
        1/np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1)),
        1/np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0)),
        1/np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0)),
    ]
    # No probabilities provided
    primal_value, _ = state_exclusion(vectors=states, probs=None, primal_dual="primal")
    np.testing.assert_equal(np.isclose(primal_value, 0), True)

    dual_value, _ = state_exclusion(vectors=states, probs=None, primal_dual="dual")
    np.testing.assert_equal(np.isclose(dual_value, 0), True)

    # Probabilities provided
    primal_value, _ = state_exclusion(vectors=states, probs=[1/4, 1/4, 1/4, 1/4], primal_dual="primal")
    np.testing.assert_equal(np.isclose(primal_value, 0), True)

    dual_value, _ = state_exclusion(vectors=states, probs=[1/4, 1/4, 1/4, 1/4], primal_dual="dual")
    np.testing.assert_equal(np.isclose(dual_value, 0), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
