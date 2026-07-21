"""Tests for cones._utils helpers."""

from toqito.cones._utils import _is_power_of_two


def test_is_power_of_two():
    """Powers of two are detected; non-positive and non-powers are rejected."""
    assert _is_power_of_two(1) is True
    assert _is_power_of_two(2) is True
    assert _is_power_of_two(8) is True
    assert _is_power_of_two(6) is False
    assert _is_power_of_two(0) is False
    assert _is_power_of_two(-4) is False
