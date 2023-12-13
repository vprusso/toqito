"""Test sk_vector_norm."""
import numpy as np
import pytest

from toqito.state_props import sk_vector_norm
from toqito.states import max_entangled


@pytest.mark.parametrize("n, k", [(4, 1), (4, 2), (5, 2)])
def test_sk_norm_maximally_entagled_state(n, k):
    """The S(k)-norm of the maximally entagled state."""
    v_vec = max_entangled(n)
    res = sk_vector_norm(v_vec, k=k)
    assert np.isclose(res, np.sqrt(k / n))


@pytest.mark.parametrize("n, k, dim", [(4, 2, 1), (4, 2, [1]), (5, 2, 1), (5, 2, [1])])
def test_sk_norm_maximally_entagled_state_with_dim(n, k, dim):
    """The S(k)-norm of the maximally entagled state where k> input_dim."""
    v_vec = max_entangled(n)
    res = sk_vector_norm(v_vec, k=k, dim=dim)
    assert np.isclose(res, 1.0)
