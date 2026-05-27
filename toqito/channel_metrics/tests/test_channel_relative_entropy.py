"""Tests for channel_relative_entropy."""

import numpy as np
import pytest

from toqito.channel_metrics.channel_relative_entropy import channel_relative_entropy
from toqito.channels import depolarizing, pauli_channel
from toqito.perms import swap_operator


def _dense(mat):
    """Convert a sparse or matrix-like channel representation to an ndarray."""
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def test_identical_channels_zero():
    """Identical channels should give zero in both bounds and mean modes."""
    choi = depolarizing(2, 1)
    hamiltonian = np.zeros((2, 2), dtype=complex)

    lower, upper = channel_relative_entropy(
        choi,
        choi,
        in_dim=2,
        epsilon_dec=0.2,
        hamiltonian=hamiltonian,
        energy=0.3,
        mean=False,
    )
    avg = channel_relative_entropy(
        choi,
        choi,
        in_dim=2,
        epsilon_dec=0.2,
        hamiltonian=hamiltonian,
        energy=0.3,
        mean=True,
    )

    assert lower == 0
    assert upper == 0
    assert avg == 0


@pytest.mark.slow
def test_bounds_order_for_distinct_channels():
    """Distinct channels should produce ordered lower and upper bounds."""
    hamiltonian = np.zeros((2, 2), dtype=complex)
    lower, upper = channel_relative_entropy(
        depolarizing(2, 1),
        depolarizing(2, 0.2),
        in_dim=2,
        epsilon_dec=0.2,
        hamiltonian=hamiltonian,
        energy=0.3,
        mean=False,
    )

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert upper >= lower


def test_raises_mismatched_choi_shapes():
    """Mismatched Choi dimensions should raise."""
    hamiltonian = np.zeros((2, 2), dtype=complex)
    with pytest.raises(ValueError, match="equal dimension"):
        channel_relative_entropy(
            depolarizing(2, 0.2),
            depolarizing(4, 0.2),
            in_dim=2,
            epsilon_dec=0.2,
            hamiltonian=hamiltonian,
            energy=0.0,
        )


def test_raises_non_square_choi():
    """Non-square Choi matrices should raise."""
    bad = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)
    hamiltonian = np.zeros((1, 1), dtype=complex)

    with pytest.raises(ValueError, match="must be square"):
        channel_relative_entropy(
            bad, bad, in_dim=1, epsilon_dec=0.2, hamiltonian=hamiltonian, energy=0.0
        )


def test_raises_bad_in_dim():
    """Choi dimensions not divisible by in_dim should raise."""
    hamiltonian = np.zeros((3, 3), dtype=complex)
    with pytest.raises(ValueError, match="divisible by in_dim"):
        channel_relative_entropy(
            depolarizing(2, 0.2),
            depolarizing(2, 0.4),
            in_dim=3,
            epsilon_dec=0.2,
            hamiltonian=hamiltonian,
            energy=0.0,
        )


def test_channel_1_must_be_quantum_channel():
    """A non-quantum first argument should raise a clear ValueError."""
    bad_channel = np.array([[1.0, 2.0, 3.0, 4.0]] * 4, dtype=complex)
    good_channel = np.eye(4, dtype=complex) / 2
    hamiltonian = np.zeros((2, 2), dtype=complex)

    with pytest.raises(ValueError, match="channel_1 is a quantum channel"):
        channel_relative_entropy(
            bad_channel,
            good_channel,
            in_dim=2,
            epsilon_dec=0.2,
            hamiltonian=hamiltonian,
            energy=0.0,
        )


def test_channel_2_must_be_completely_positive():
    """A non-CP second argument should raise a clear ValueError."""
    good_channel = np.eye(4, dtype=complex) / 2
    non_cp_channel = swap_operator(2).astype(complex)
    hamiltonian = np.zeros((2, 2), dtype=complex)

    with pytest.raises(ValueError, match="channel_2 is completely positive"):
        channel_relative_entropy(
            good_channel,
            non_cp_channel,
            in_dim=2,
            epsilon_dec=0.2,
            hamiltonian=hamiltonian,
            energy=0.0,
        )


def test_raises_bad_hamiltonian_shape():
    """Hamiltonian must match the input dimension."""
    with pytest.raises(ValueError, match="Hamiltonian"):
        channel_relative_entropy(
            depolarizing(2, 0.2),
            depolarizing(2, 0.4),
            in_dim=2,
            epsilon_dec=0.2,
            hamiltonian=np.zeros((3, 3), dtype=complex),
            energy=0.0,
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("param_p", "expected_mean"),
    [
        (0.010, 2.920),
        (0.013, 2.759),
        (0.016, 2.633),
        (0.019, 2.529),
        (0.022, 2.440),
        (0.026, 2.364),
        (0.029, 2.296),
        (0.032, 2.235),
        (0.035, 2.180),
        (0.038, 2.130),
        (0.041, 2.084),
        (0.044, 2.043),
        (0.047, 2.003),
        (0.050, 1.967),
        (0.053, 1.932),
        (0.057, 1.898),
        (0.060, 1.865),
        (0.063, 1.834),
        (0.066, 1.803),
        (0.069, 1.776),
        (0.072, 1.751),
        (0.075, 1.726),
        (0.078, 1.702),
        (0.081, 1.681),
        (0.084, 1.660),
        (0.088, 1.639),
        (0.091, 1.619),
        (0.094, 1.599),
        (0.097, 1.582),
        (0.100, 1.570),
    ],
)
def test_channel_relative_entropy_paper_example(param_p: float, expected_mean: float):
    """Mean estimate matches the paper's qubit dephasing-vs-depolarizing example."""
    # Paper example:
    # N_deph(rho) = 0.4 rho + 0.6 sigma_z rho sigma_z
    # M_dep(rho) = (1 - 3p/4) rho + p/4 (X rho X + Y rho Y + Z rho Z)
    channel_1 = _dense(pauli_channel(np.array([0.4, 0.0, 0.0, 0.6])))
    channel_2 = _dense(
        pauli_channel(
            np.array([1 - 3 * param_p / 4, param_p / 4, param_p / 4, param_p / 4])
        )
    )
    hamiltonian = np.zeros((2, 2), dtype=complex)

    lower, upper = channel_relative_entropy(
        channel_1,
        channel_2,
        in_dim=2,
        epsilon_dec=1e-2,
        hamiltonian=hamiltonian,
        energy=0.0,
        mean=False,
    )
    avg = channel_relative_entropy(
        channel_1,
        channel_2,
        in_dim=2,
        epsilon_dec=1e-2,
        hamiltonian=hamiltonian,
        energy=0.0,
        mean=True,
    )

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert upper >= lower
    assert avg == pytest.approx(expected_mean, abs=2e-2)
