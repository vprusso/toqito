"""Test state_exclusion."""
import numpy as np

from toqito.state_opt import state_exclusion
from toqito.states import bell


def test_conclusive_state_exclusion_one_state_vec():
    """Conclusive state exclusion for single vector state."""
    value, _ = state_exclusion(vectors=[bell(0)], probs=None)
    np.testing.assert_equal(np.isclose(value, 1), True)


def test_conclusive_state_exclusion_three_state():
    """Conclusive state exclusion for three Bell state vectors."""
    value, _ = state_exclusion(vectors=[bell(0), bell(1), bell(2)], probs=[1/3, 1/3, 1/3])
    np.testing.assert_equal(np.isclose(value, 0), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
