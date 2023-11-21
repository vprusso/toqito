"""Test mutually_unbiased_basis."""
import numpy as np
import pytest

from toqito.states import mutually_unbiased_basis
from toqito.state_props import is_mutually_unbiased_basis


@pytest.mark.parametrize("dim", [2, 3, 5, 7])
def test_mutually_unbiased_basis(dim):
    np.testing.assert_equal(is_mutually_unbiased_basis(mutually_unbiased_basis(dim)), True)

