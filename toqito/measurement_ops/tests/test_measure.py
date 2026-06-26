"""Test suite for the measure function."""

import numpy as np
import pytest

from toqito.measurement_ops.measure import measure


@pytest.mark.parametrize(
    "state, measurement, state_update, expected",
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
def test_measure_single_operator(state, measurement, state_update, expected):
    """Test cases for a single operator."""
    result = measure(state, measurement, state_update=state_update)
    if state_update:
        exp_prob, exp_post = expected
        res_prob, res_post = result
        assert np.isclose(res_prob, exp_prob, atol=1e-7)
        np.testing.assert_allclose(res_post, exp_post, rtol=1e-7)
    else:
        assert np.isclose(result, expected, atol=1e-7)


@pytest.mark.parametrize(
    "state, measurements, state_update, expected",
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
def test_measure_multiple_operators(state, measurements, state_update, expected):
    """Test cases for multiple operators."""
    result = measure(state, measurements, state_update=state_update)
    if state_update:
        for (res_prob, res_post), (exp_prob, exp_post) in zip(result, expected):
            assert np.isclose(res_prob, exp_prob, atol=1e-7)
            np.testing.assert_allclose(res_post, exp_post, rtol=1e-7)
    else:
        for res, exp in zip(result, expected):
            assert np.isclose(res, exp, atol=1e-7)


@pytest.mark.parametrize(
    "state, measurement, state_update",
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
def test_measure_zero_operator(state, measurement, state_update):
    """Test cases when the measurement operator yields zero probability."""
    result = measure(state, measurement, state_update=state_update)
    if state_update:
        prob, post_state = result
        assert np.isclose(prob, 0.0, atol=1e-7)
        np.testing.assert_allclose(post_state, np.zeros_like(state), rtol=1e-7)
    else:
        assert np.isclose(result, 0.0, atol=1e-7)


@pytest.mark.parametrize(
    "state_update, expected",
    [
        # A complete measurement {P_0, P_1} on |0><0|: the P_1 outcome has zero probability.
        (False, [1.0, 0.0]),
        (
            True,
            [
                (1.0, np.array([[1, 0], [0, 0]])),
                (0.0, np.zeros((2, 2))),
            ],
        ),
    ],
)
def test_measure_list_with_zero_probability_outcome(state_update, expected):
    """A valid complete measurement with a zero-probability outcome returns zero (and a zero post-state)."""
    state = np.array([[1, 0], [0, 0]])
    measurements = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]
    result = measure(state, measurements, state_update=state_update)

    if state_update:
        for (res_p, res_post), (exp_p, exp_post) in zip(result, expected):
            assert np.isclose(res_p, exp_p, atol=1e-7)
            np.testing.assert_allclose(res_post, exp_post, rtol=1e-7)
    else:
        for res, exp in zip(result, expected):
            assert np.isclose(res, exp, atol=1e-7)


@pytest.mark.parametrize("state_update", [True, False])
def test_measure_completeness_failure(state_update):
    """A list of operators violating completeness raises ValueError regardless of state_update."""
    state = np.array([[0.5, 0.5], [0.5, 0.5]])
    measurements = [np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])]
    with pytest.raises(ValueError, match="Kraus operators do not satisfy completeness relation"):
        measure(state, measurements, state_update=state_update)


@pytest.mark.parametrize("state_update", [True, False])
def test_measure_incomplete_list_with_zero_operator_raises(state_update):
    """An incomplete list (e.g. a zero operator and a projector) is rejected even though one outcome is zero."""
    state = np.array([[0.5, 0.5], [0.5, 0.5]])
    measurements = [np.zeros((2, 2)), np.array([[1, 0], [0, 0]])]
    with pytest.raises(ValueError, match="Kraus operators do not satisfy completeness relation"):
        measure(state, measurements, state_update=state_update)


def test_measure_empty_list_raises():
    """An empty measurement list is rejected."""
    state = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="At least one measurement operator is required"):
        measure(state, [])


def test_measure_invalid_density_matrix():
    """Test that passing a non-density matrix raises ValueError."""
    invalid_state = np.array([[1, 1], [1, 1]])  # Not positive semidefinite
    measurement = np.array([[1, 0], [0, 0]])

    with pytest.raises(ValueError, match="Input must be a valid density matrix"):
        measure(invalid_state, measurement)
