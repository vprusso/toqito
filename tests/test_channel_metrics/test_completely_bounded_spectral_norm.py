"""Tests for completely_bounded_spectral_norm."""
import numpy as np

from toqito.channel_metrics import (
    completely_bounded_spectral_norm,
    completely_bounded_trace_norm,
)
from toqito.channels import dephasing
from toqito.channel_ops import dual_channel


def test_dual_is_cb_trace_norm():
    phi = dephasing(2)
    np.testing.assert_equal(
        completely_bounded_spectral_norm(phi), completely_bounded_trace_norm(dual_channel(phi))
    )


if __name__ == "__main__":
    np.testing.run_module_suite()
