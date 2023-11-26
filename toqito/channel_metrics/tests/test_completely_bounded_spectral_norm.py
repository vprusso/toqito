"""Tests for completely_bounded_spectral_norm."""
import numpy as np

from toqito.channel_metrics import (
    completely_bounded_spectral_norm,
    completely_bounded_trace_norm,
)
from toqito.channel_ops import dual_channel
from toqito.channels import dephasing


def test_dual_is_cb_trace_norm():
    """Test CB trace norm is equal to CB Spectral norm."""
    phi = dephasing(2)
    np.testing.assert_equal(
        completely_bounded_spectral_norm(phi), completely_bounded_trace_norm(dual_channel(phi))
    )
