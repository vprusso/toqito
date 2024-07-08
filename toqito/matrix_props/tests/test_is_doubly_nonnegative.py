"""Tests for doubly nonnegative matrix check function."""

import numpy as np
import pytest

from toqito.matrix_props import is_doubly_nonnegative


def test_identity():
    """Check an identity matrix is nonnegative as expected."""
    assert is_doubly_nonnegative(np.identity(3))
