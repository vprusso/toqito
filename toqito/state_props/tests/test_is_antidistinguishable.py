"""Test is_antidistinguishable."""

import pytest

from toqito.state_props import is_antidistinguishable
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
