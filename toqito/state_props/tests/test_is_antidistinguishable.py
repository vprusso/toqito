"""Test is_antidistinguishable."""

import pytest

from toqito.matrix_ops import vectors_from_gram_matrix
from toqito.rand import random_circulant_gram_matrix
from toqito.state_props import is_antidistinguishable, is_antidistinguishable_circulant
from toqito.states import bell, trine


@pytest.mark.parametrize(
    "states",
    [
        # The Bell states are known to be antidistinguishable.
        ([bell(0), bell(1), bell(2), bell(3)]),
        # The trine states are known to be antidistinguishable.
        ([trine()[0], trine()[1], trine()[2]]),
    ],
)
def test_is_antidistinguishable(states):
    """Test function works as expected for a valid input."""
    assert is_antidistinguishable(states)


def test_is_antidistinguishable_benchmark(benchmark):
    """Benchmark for is_antidistinguishable function."""
    gram_matrix = random_circulant_gram_matrix(16, seed=42)
    states = vectors_from_gram_matrix(gram_matrix)
    benchmark(is_antidistinguishable_circulant(states))
