"""Tests random_psd"""

import numpy as np
import pytest

from toqito.rand import random_psd


@pytest.mark.parametrize(
    "dim",
    "iscplx",
    [
        # Test different sized matrices
        (2, True),
        (2, False),
        (3, True),
        (3, False),
        (4, True),
        (4, False),
        (5, True),
        (5, False),
    ],
)
def test_random_psd(dim, is_cplx):
    """Test the random_psd function"""
    # Generate the matrix
    psd = random_psd(dim, is_cplx)

    # Check size is right
    assert len(psd) == len(psd[0]) == dim

    # Check if the matrix is positive semidefinite
    eigenvalues = np.linalg.eigvals(psd)
    assert np.all(eigenvalues >= 0)
