"""Test in_separable_ball."""

import numpy as np
import pytest

from toqito.rand import random_unitary
from toqito.state_props import in_separable_ball

random_u_mat = random_unitary(4)


@pytest.mark.parametrize(
    "rho, expected_result",
    [
        # Test matrix in separable ball returns True.
        (np.identity(4) @ np.diag(np.array([1, 1, 1, 0])) / 3 @ np.identity(4).conj().T, True),
        # Test matrix not in separable ball returns False.
        (random_u_mat @ np.diag(np.array([1.01, 1, 0.99, 0])) / 3 @ random_u_mat.conj().T, False),
        # Test for case when trace of matrix is less than the largest dim."""
        (np.zeros((4, 4)), False),
        # Test eigenvalues of matrix not in separable ball returns False.
        (
            np.linalg.eigvalsh(random_u_mat @ np.diag(np.array([1.01, 1, 0.99, 0])) / 3 @ random_u_mat.conj().T),
            False,
        ),
    ],
)
def test_in_separable_ball(rho, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(in_separable_ball(rho), expected_result)
