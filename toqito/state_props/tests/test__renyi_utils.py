"""Direct tests for the private Rényi helpers in _renyi_utils."""

import numpy as np

from toqito.state_props._renyi_utils import support_projector


def test_support_projector_of_zero_matrix_is_zero():
    """support_projector on a PSD matrix with no support (all-zero) returns the zero matrix."""
    result = support_projector(np.zeros((3, 3)))
    np.testing.assert_allclose(result, np.zeros((3, 3)))


def test_support_projector_of_subtol_eigvals_is_zero():
    """If all eigenvalues are below tol, support_projector returns the zero matrix."""
    mat = 1e-14 * np.eye(2)
    result = support_projector(mat, tol=1e-12)
    np.testing.assert_allclose(result, np.zeros((2, 2)))
