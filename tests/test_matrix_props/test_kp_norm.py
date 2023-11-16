"""Tests for the S(k)-norm of a matrix."""
# Tests in this file, follow the discussion of the characteristics of kp_norm at https://qetlab.com/kpNorm

import re

import numpy as np
import pytest

from toqito.matrix_props import kp_norm, trace_norm
from toqito.states import bell


def test_operator_norm():
    """When (k=1, p= Inf)the kp_norm(vector) is the same as the trace norm."""
    calculated_kp_norm = kp_norm(bell(0), 1, np.inf)
    expected_kp_norm = trace_norm(bell(0))
    assert calculated_kp_norm == expected_kp_norm


def test_frobenius_norm():
    """When p=2 and k is greater than or equal to one of the input matrice's
    dimensions, the value calculated is the frobenius norm."""
    input_mat = np.random.rand(5, 4)
    k = min(input_mat.shape)
    p = 2
    calculated_value = kp_norm(input_mat, k, p)
    expected_value = np.linalg.norm(input_mat, ord="fro")
    assert calculated_value == expected_value


def test_no_default_kp_values():
    """kp_norm does not have any default values for k or p."""
    with pytest.raises(
        TypeError, match=re.escape("kp_norm() missing 2 required positional arguments: 'k' and 'p'")
    ):
        kp_norm(bell(0))  # pylint: disable=no-value-for-parameter
