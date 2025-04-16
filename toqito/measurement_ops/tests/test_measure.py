"""Test suite for the measure function."""

import numpy as np
import pytest

from toqito.measurement_ops.measure import measure


# Test cases for a single operator.
@pytest.mark.parametrize(
    "state, measurement, update, expected",
    [
        # Single operator, no update.
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[1, 0], [0, 0]]),
            False,
            0.5,
        ),
        # Single operator, with update.
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[1, 0], [0, 0]]),
            True,
            (0.5, np.array([[1, 0], [0, 0]])),
        ),
    ],
)
def test_measure_single_operator(state, measurement, update, expected):
    result = measure(state, measurement, update=update)
    if update:
        exp_prob, exp_post = expected
        res_prob, res_post = result
        assert np.isclose(res_prob, exp_prob, atol=1e-7)
        np.testing.assert_allclose(res_post, exp_post, rtol=1e-7)
    else:
        assert np.isclose(result, expected, atol=1e-7)


# Test cases for multiple operators.
@pytest.mark.parametrize(
    "state, measurements, update, expected",
    [
        # Multiple operators, no update.
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])],
            False,
            [0.5, 0.5],
        ),
        # Multiple operators, with update.
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])],
            True,
            [
                (0.5, np.array([[1, 0], [0, 0]])),
                (0.5, np.array([[0, 0], [0, 1]])),
            ],
        ),
        # Multiple operators given as a tuple.
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])),
            False,
            [0.5, 0.5],
        ),
    ],
)
def test_measure_multiple_operators(state, measurements, update, expected):
    result = measure(state, measurements, update=update)
    if update:
        for (res_prob, res_post), (exp_prob, exp_post) in zip(result, expected):
            assert np.isclose(res_prob, exp_prob, atol=1e-7)
            np.testing.assert_allclose(res_post, exp_post, rtol=1e-7)
    else:
        for res, exp in zip(result, expected):
            assert np.isclose(res, exp, atol=1e-7)


# Test cases when the measurement operator yields zero probability.
@pytest.mark.parametrize(
    "state, measurement, update",
    [
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.zeros((2, 2)),
            False,
        ),
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.zeros((2, 2)),
            True,
        ),
    ],
)
def test_measure_zero_operator(state, measurement, update):
    result = measure(state, measurement, update=update)
    if update:
        prob, post_state = result
        assert np.isclose(prob, 0.0, atol=1e-7)
        np.testing.assert_allclose(post_state, np.zeros_like(state), rtol=1e-7)
    else:
        assert np.isclose(result, 0.0, atol=1e-7)


# Test that a list of operators that does not satisfy the completeness relation raises ValueError.
@pytest.mark.parametrize(
    "state, measurements",
    [
        (
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            [np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])],
        ),
    ],
)
def test_measure_completeness_failure(state, measurements):
    with pytest.raises(ValueError, match="Kraus operators do not satisfy completeness relation"):
        measure(state, measurements, update=True)
