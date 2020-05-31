"""Tests for trace_norm."""
import numpy as np

from toqito.state_metrics import trace_norm
from toqito.states import basis


def test_trace_norm():
    """Test trace norm."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec * u_vec.conj().T

    res = trace_norm(rho)
    _, singular_vals, _ = np.linalg.svd(rho)
    expected_res = float(np.sum(singular_vals))

    np.testing.assert_equal(np.isclose(res, expected_res), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
