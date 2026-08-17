"""Tests for the S(k)-norm of a matrix."""
# Tests in this file, follow the discussion of the characteristics of kp_norm at https://qetlab.com/kpNorm

import re

import numpy as np
import pytest

from toqito.matrix_props import kp_norm
from toqito.rand import random_unitary
from toqito.states import bell

# A single fixed unitary, drawn once. Two separate `random_unitary(5)` calls would compare
# `kp_norm` of one matrix against the Frobenius norm of a different one, which passes only
# because that norm is the same for every 5x5 unitary. Seeding also keeps the generated test
# ID stable, which parallel collection requires.
_UNITARY_5 = random_unitary(5, seed=42)


@pytest.mark.parametrize(
    "vector, k, p, norm_to_compare",
    [
        # When (k=1, p= Inf)the kp_norm(vector) is the same as the trace norm (the 1-norm).
        (bell(0), 1, np.inf, 1),
        # When p=2 and k is greater than or equal to one of the input matrix dimensions, the value calculated is
        # the frobenius norm.
        (_UNITARY_5, 5, 2, np.linalg.norm(_UNITARY_5, "fro")),
    ],
)
def test_kp_norm(vector, k, p, norm_to_compare):
    """Test function works as expected for a valid input."""
    calculated_kp_norm = kp_norm(vector, k, p)
    assert calculated_kp_norm == pytest.approx(norm_to_compare)


def test_no_default_kp_values():
    """Test kp_norm does not have any default values for k or p."""
    with pytest.raises(TypeError, match=re.escape("kp_norm() missing 2 required positional arguments: 'k' and 'p'")):
        kp_norm(bell(0))
