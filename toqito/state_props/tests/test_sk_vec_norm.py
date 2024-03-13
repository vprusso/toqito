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


@pytest.mark.parametrize("n, k, expected_result", [(4, 2, 0.7), (5, 2, 0.63)])
def test_sk_norm_maximally_entagled_state_with_none_dim(n, k, expected_result):
    """The S(k)-norm of the maximally entagled state dim = None."""
    v_vec = max_entangled(n)
    res = sk_vector_norm(v_vec, k=k, dim=None)
    assert np.isclose(res, expected_result, atol=0.01)
