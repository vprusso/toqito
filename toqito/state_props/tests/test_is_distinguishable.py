"""Test is_distinguishable."""

import pytest

from toqito.state_props import is_distinguishable
from toqito.states import bell, trine


@pytest.mark.parametrize(
    "states, probs, is_dist",
    [
        # The Bell states are known to be distinguishable.
        ([bell(0), bell(1), bell(2), bell(3)], [1 / 4, 1 / 4, 1 / 4, 1 / 4], True),
        # The trine states are known to not be distinguishable.
        ([trine()[0], trine()[1], trine()[2]], [1 / 4, 1 / 4, 1 / 4, 1 / 4], False),
    ],
)
def test_is_distinguishable(states, probs, is_dist):
    """Test function works as expected for a valid input."""
    assert is_distinguishable(states, probs) == is_dist
