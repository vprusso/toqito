"""Test is_pseudo_hermitian."""

import numpy as np
import pytest

from toqito.matrix_props import is_pseudo_hermitian


@pytest.mark.parametrize(
    "mat, signature, expected",
    [
        # Pseudo-Hermitian matrix example
        (
            np.array([[1, 1 + 1j], [-1 + 1j, -1]]),
            np.array([[1, 0], [0, -1]]),
            True,
        ),
        # Non pseudo-Hermitian matrix
        (
            np.array([[1, 1j], [-1j, 1]]),
            np.array([[1, 0], [0, -1]]),
            False,
        ),
        # Non-square matrix should return False
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 0], [0, -1]]),
            False,
        ),
        # Mismatched dimensions should return False
        (
            np.array([[1, 1j], [-1j, 1]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            False,
        ),
    ],
)
def test_is_pseudo_hermitian(mat, signature, expected):
    """Test that is_pseudo_hermitian gives correct boolean value on valid inputs."""
    np.testing.assert_equal(is_pseudo_hermitian(mat, signature), expected)


def test_is_pseudo_hermitian_value_error():
    """Signature matrix must be Hermitian and invertible."""
    mat = np.array([[1, 1j], [-1j, 1]])

    # Non-Hermitian signature matrix
    non_hermitian_signature = np.array([[1, 2], [3, 4]])
    np.testing.assert_raises_regex(
        ValueError,
        "Signature not hermitian matrix.",
        is_pseudo_hermitian,
        mat,
        non_hermitian_signature,
    )

    # Singular signature matrix (not invertible)
    singular_signature = np.array([[1, 1], [1, 1]])
    np.testing.assert_raises_regex(
        ValueError,
        "Signature is not invertible.",
        is_pseudo_hermitian,
        mat,
        singular_signature,
    )
