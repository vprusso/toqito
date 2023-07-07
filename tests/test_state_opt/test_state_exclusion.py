"""Test state_exclusion."""
import numpy as np

from toqito.state_opt import state_exclusion
from toqito.matrices import standard_basis


def test_conclusive_state_exclusion_one_state_vec():
    """Conclusive state exclusion for single vector state."""
    e_0, e_1 = standard_basis(2)
    trine = [
        e_0,
        1/2 * (-e_0 + np.sqrt(3) * e_1),
        -1/2 * (e_0 + np.sqrt(3) * e_1),
    ]
    # No probabilities provided
    primal_value, _ = state_exclusion(vectors=trine, probs=None, primal_dual="primal")
    np.testing.assert_equal(np.isclose(primal_value, 0), True)

    dual_value, _ = state_exclusion(vectors=trine, probs=None, primal_dual="dual")
    np.testing.assert_equal(np.isclose(dual_value, 0), True)

    # Probabilities provided
    primal_value, _ = state_exclusion(vectors=trine, probs=[1/3, 1/3, 1/3], primal_dual="primal")
    np.testing.assert_equal(np.isclose(primal_value, 0), True)

    dual_value, _ = state_exclusion(vectors=trine, probs=[1/3, 1/3, 1/3], primal_dual="dual")
    np.testing.assert_equal(np.isclose(dual_value, 0), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
