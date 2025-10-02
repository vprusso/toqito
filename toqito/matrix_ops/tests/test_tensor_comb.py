"""Test tensor_comb."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor_comb


@pytest.mark.parametrize(
    "states, k, mode, expected_comb_keys, expected_comb",
    [
        # Injective mode with k = 1.
        (
            [np.array([1, 0]), np.array([0, 1])],
            1,
            "injective",
            [(0,), (1,)],
            {(0,): np.array([1, 0]), (1,): np.array([0, 1])},
        ),
        # Non-injective mode with k = 1.
        (
            [np.array([1, 0]), np.array([0, 1])],
            1,
            "non-injective",
            [(0,), (1,)],
            {(0,): np.array([1, 0]), (1,): np.array([0, 1])},
        ),
        # Non-injective mode with k = 2.
        (
            [np.array([1, 0]), np.array([0, 1])],
            2,
            "non-injective",
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            {
                (0, 0): np.array([1, 0, 0, 0]),
                (0, 1): np.array([0, 1, 0, 0]),
                (1, 0): np.array([0, 0, 1, 0]),
                (1, 1): np.array([0, 0, 0, 1]),
            },
        ),
        # diagonal mode with k = 3.
        (
            [np.array([1, 0]), np.array([0, 1])],
            3,
            "diagonal",
            [(0, 0, 0), (1, 1, 1)],
            {(0, 0, 0): np.array([1, 0, 0, 0, 0, 0, 0, 0]), (1, 1, 1): np.array([0, 0, 0, 0, 0, 0, 0, 1])},
        ),
        # diagonal mode with k = 2.
        (
            [np.array([1, 0]), np.array([0, 1])],
            2,
            "diagonal",
            [(0, 0), (1, 1)],
            {
                (0, 0): np.array([1, 0, 0, 0]),
                (1, 1): np.array([0, 0, 0, 1]),
            },
        ),
    ],
)
def test_tensor_comb(states, k, mode, expected_comb_keys, expected_comb):
    """Test the tensor_comb function."""
    result = tensor_comb(states, k, mode=mode, density_matrix=False)
    assert len(result) == len(expected_comb_keys)
    for key in expected_comb_keys:
        calc_res = result[key]
        expected_res = expected_comb[key]
        assert (calc_res == expected_res).all()


@pytest.mark.parametrize(
    "states, k, mode, expected_comb_keys, expected_rho",
    [
        # Injective mode with k = 1.
        (
            [np.array([1, 0]), np.array([0, 1])],
            1,
            "injective",
            [(0,), (1,)],
            {(0,): np.array([[1, 0], [0, 0]]), (1,): np.array([[0, 0], [0, 1]])},
        ),
        # Non-injective mode with k = 1.
        (
            [np.array([1, 0]), np.array([0, 1])],
            1,
            "non-injective",
            [(0,), (1,)],
            {(0,): np.array([[1, 0], [0, 0]]), (1,): np.array([[0, 0], [0, 1]])},
        ),
        # Non-injective mode with k = 2.
        (
            [np.array([1, 0]), np.array([0, 1])],
            2,
            "non-injective",
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            {
                (0, 0): np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                (0, 1): np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                (1, 0): np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
                (1, 1): np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
            },
        ),
        # Non-injective mode with k = 3.
        (
            [np.array([1, 0]), np.array([0, 1])],
            3,
            "diagonal",
            [(0, 0, 0), (1, 1, 1)],
            {
                (0, 0, 0): np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                (1, 1, 1): np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            },
        ),
        # diagonal mode with k = 2.
        (
            [np.array([1, 0]), np.array([0, 1])],
            2,
            "diagonal",
            [(0, 0), (1, 1)],
            {
                (0, 0): np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                (1, 1): np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                    ]
                ),
            },
        ),
    ],
)
def test_tensor_comb_density_matrix(states, k, mode, expected_comb_keys, expected_rho):
    """Test the tensor_comb function when it returns the density matrix."""
    result = tensor_comb(states, k, mode=mode, density_matrix=True)
    assert len(result) == len(expected_comb_keys)
    for key in expected_comb_keys:
        calc_res = result[key]
        expected_res = expected_rho[key]
        assert (calc_res == expected_res).all()


@pytest.mark.parametrize(
    "test_input, test_k, test_density_matrix, test_mode, expected_msg",
    [
        # k is greater than len(states) for injective mode
        (
            [np.array([1, 0]), np.array([0, 1])],
            3,
            False,
            "injective",
            "k must be less than or equal to the number of states for injective sequences.",
        ),
        # invalid mode option
        (
            [np.array([1, 0]), np.array([0, 1])],
            2,
            False,
            "invalid",
            "mode must be injective, non-injective, or diagonal.",
        ),
        # empty input
        ([], 2, False, "injective", "Input list of states cannot be empty."),
    ],
)
def test_raised_errors(test_input, test_k, test_density_matrix, test_mode, expected_msg):
    """Test function raises error as expected for invalid inputs."""
    with pytest.raises(ValueError, match=expected_msg):
        tensor_comb(test_input, test_k, mode=test_mode, density_matrix=test_density_matrix)
