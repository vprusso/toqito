"""Tests for channel measured relative entropy."""

import numpy as np

from toqito.channel_metrics.channel_measured_relative_entropy import channel_measured_relative_entropy
from toqito.channel_ops import partial_channel
from toqito.channels import depolarizing, partial_trace
from toqito.measurement_ops import measure
from toqito.rand import random_density_matrix, random_povm


def _classical_relative_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Classical KL divergence D(p||q). Convention: if exists i with p_i > 0 and q_i == 0, return +inf."""
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)

    if np.any((p > 0) & (q == 0)):
        return float("inf")

    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def _monto_carlo_lower_bound(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    in_dim: int,
    out_dim: int,
    hamiltonian: np.ndarray,
    energy: float,
    seed: int,
    n_inputs: int = 40,
    n_povms: int = 80,
    povm_num_outputs: int | None = None,
) -> float:
    """Monte Carlo lower bound on the energy-constrained measured relative entropy of channels.

    We sample feasible input states rho_RA satisfying Tr[H rho_A] <= E and for each
    we sample POVMs on RB to get classical distributions and compute classical KL.
    """
    rng = np.random.default_rng(seed)

    # Make Hamiltonian Hermitian
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2

    dR = in_dim
    dim_RA = dR * in_dim
    dim_RB = dR * out_dim

    if povm_num_outputs is None:
        povm_num_outputs = dim_RB

    best = -np.inf

    for _ in range(n_inputs):
        # Sample a random state on RA
        rho_RA = random_density_matrix(dim_RA, distance_metric="haar")

        # Compute reduced state on A
        rho_A = partial_trace(rho_RA, sys=[0], dim=[dR, in_dim])

        exp_energy = float(np.real(np.trace(hamiltonian @ rho_A)))
        if exp_energy > energy + 1e-10:
            continue

        # Apply channels
        rho_RB_1 = partial_channel(rho_RA, channel_1)
        rho_RB_2 = partial_channel(rho_RA, channel_2)

        # Normalize outputs so they are valid density matrices
        rho_RB_1 /= np.trace(rho_RB_1)
        rho_RB_2 /= np.trace(rho_RB_2)

        best_for_rho = -np.inf

        for _ in range(n_povms):
            povms = random_povm(
                dim_RB,
                num_inputs=1,
                num_outputs=povm_num_outputs,
                seed=rng.integers(2**32 - 1),
            )
            povm = list(povms[0])

            p = np.asarray(measure(rho_RB_1, povm), dtype=float)
            q = np.asarray(measure(rho_RB_2, povm), dtype=float)

            # Numerical cleanup
            p = np.maximum(p, 0.0)
            q = np.maximum(q, 0.0)
            if p.sum() <= 0 or q.sum() <= 0:
                continue
            p /= p.sum()
            q /= q.sum()

            kl = _classical_relative_entropy(p, q)
            if np.isfinite(kl):
                best_for_rho = max(best_for_rho, kl)

        best = max(best, best_for_rho)

    return best


def test_sdp_lower_bound():
    """Test that the SDP result is greater than the Monte Carlo lower bound."""
    in_dim = 2
    out_dim = 2

    channel_1 = np.eye(in_dim * out_dim, dtype=complex) / out_dim
    channel_2 = depolarizing(in_dim, 0.25)

    hamiltonian = np.zeros((in_dim, in_dim), dtype=complex)
    energy = 1e6

    sdp = channel_measured_relative_entropy(
        channel_1,
        channel_2,
        in_dim=in_dim,
        m=6,
        k=6,
        hamiltonian=hamiltonian,
        energy=energy,
    )

    lb = _monto_carlo_lower_bound(
        channel_1,
        channel_2,
        in_dim=in_dim,
        out_dim=out_dim,
        hamiltonian=hamiltonian,
        energy=energy,
        seed=123,
        n_inputs=35,
        n_povms=60,
    )

    assert np.isfinite(sdp)
    assert np.real(sdp) + 5e-2 >= lb


def test_energy_constraint():
    """Test for energy constraint."""
    in_dim = 2
    out_dim = 2

    channel_1 = np.eye(in_dim * out_dim, dtype=complex) / out_dim
    channel_2 = depolarizing(in_dim, 0.35)

    hamiltonian = np.diag([0.0, 1.0]).astype(complex)
    energy = 0.0

    sdp = channel_measured_relative_entropy(
        channel_1,
        channel_2,
        in_dim=in_dim,
        m=6,
        k=6,
        hamiltonian=hamiltonian,
        energy=energy,
    )

    lb = _monto_carlo_lower_bound(
        channel_1,
        channel_2,
        in_dim=in_dim,
        out_dim=out_dim,
        hamiltonian=hamiltonian,
        energy=energy,
        seed=2020,
        n_inputs=60,
        n_povms=80,
    )

    assert np.isfinite(sdp)
    assert np.real(sdp) + 8e-2 >= lb


def test_convergence():
    """Test that SDP result converges as m and k increase, but always stays above Monte Carlo result."""
    in_dim = 2
    out_dim = 2

    channel_1 = np.eye(in_dim * out_dim, dtype=complex) / out_dim
    channel_2 = depolarizing(in_dim, 0.2)

    hamiltonian = np.zeros((in_dim, in_dim), dtype=complex)
    energy = 1e6

    lb = _monto_carlo_lower_bound(
        channel_1,
        channel_2,
        in_dim=in_dim,
        out_dim=out_dim,
        hamiltonian=hamiltonian,
        energy=energy,
        seed=7,
        n_inputs=30,
        n_povms=60,
    )

    v_small = channel_measured_relative_entropy(channel_1, channel_2, in_dim, 2, 2, hamiltonian, energy)
    v_big = channel_measured_relative_entropy(channel_1, channel_2, in_dim, 6, 6, hamiltonian, energy)

    assert np.isfinite(v_small)
    assert np.isfinite(v_big)
    assert np.real(v_small) + 8e-2 >= lb
    assert np.real(v_big) + 5e-2 >= lb
    assert np.real(v_big) + 1e-3 >= np.real(v_small)


def test_equal():
    """Test that measured relative entropy is zero when channels are identical."""
    in_dim = 2
    hamiltonian = np.zeros((in_dim, in_dim), dtype=complex)
    energy = 123.0

    channel_1 = np.eye(in_dim * in_dim, dtype=complex) / in_dim

    assert (
        channel_measured_relative_entropy(
            channel_1,
            channel_1,
            in_dim=in_dim,
            m=2,
            k=2,
            hamiltonian=hamiltonian,
            energy=energy,
        )
        == 0
    )
