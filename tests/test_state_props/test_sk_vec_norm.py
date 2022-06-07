"""Test sk_vector_norm."""
import numpy as np
import pytest

from toqito.states import max_entangled
from toqito.state_props import sk_vector_norm


@pytest.mark.parametrize("n, k", [(4, 1), (4, 2), (5, 2)])
def test_sk_norm_maximally_entagled_state(n, k):
    """The S(k)-norm of the maximally entagled state."""
    v_vec = max_entangled(n)
    res = sk_vector_norm(v_vec, k=k)

    np.testing.assert_equal(np.isclose(res, np.sqrt(k / n)), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
