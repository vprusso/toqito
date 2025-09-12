"""Test hadamard."""

import numpy as np
import pytest

from toqito.matrices import hadamard


@pytest.mark.parametrize(
    "n, expected_res",
    [
        (1, 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])),
        (2, 1 / 2 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])),
        (
            3,
            1
            / (2 ** (3 / 2))
            * np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1],
                    [1, -1, -1, 1, 1, -1, -1, 1],
                    [1, 1, 1, 1, -1, -1, -1, -1],
                    [1, -1, 1, -1, -1, 1, -1, 1],
                    [1, 1, -1, -1, -1, -1, 1, 1],
                    [1, -1, -1, 1, -1, 1, 1, -1],
                ]
            ),
        ),
    ],
)
def test_hadamard_matrix_values(n, expected_res):
    """Test for Hadamard function with specific expected matrices."""
    res = hadamard(n)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


@pytest.mark.parametrize(
    "n, expected_shape",
    [
        (4, (16, 16)),
        # You can add more shape tests here if needed
    ],
)
def test_hadamard_matrix_shape(n, expected_shape):
    """Test for Hadamard function matrix shapes."""
    res = hadamard(n)
    assert res.shape == expected_shape


@pytest.mark.parametrize("invalid_input", [-1, 0])
def test_hadamard_raises_error(invalid_input):
    """Verify function raises when an invalid parameter is provided as input."""
    with pytest.raises(ValueError, match="Provided parameter for matrix dimensions is invalid."):
        hadamard(invalid_input)
