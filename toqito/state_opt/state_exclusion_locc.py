"""Computes state exclusion under one-way LOCC via see-saw optimization."""

import cvxpy
import numpy as np

from toqito.matrix_ops import partial_trace, to_density_matrix
from toqito.rand import random_povm


def state_exclusion_locc(
    states: list[np.ndarray],
    probs: list[float] | None = None,
    dim: list[int] | None = None,
    *,
    num_alice_outcomes: int | None = None,
    reps: int = 5,
    tol: float = 1e-7,
    max_iters: int = 100,
    seed: int | None = None,
) -> float:
    r"""Compute state exclusion under one-way LOCC via a see-saw optimization.

    The *state exclusion* problem asks Bob to identify a state that was **not** sent, incurring an
    error whenever his guess coincides with the prepared state. Here the measurement is restricted
    to **one-way local operations and classical communication (LOCC)**: for a bipartite ensemble
    \(\{(p_k, \rho_k)\}\) on \(\mathcal{H}_A \otimes \mathcal{H}_B\), Alice measures her subsystem
    with a POVM \(\{A_a\}\), communicates the outcome \(a\) to Bob, who then applies a conditional
    POVM \(\{B^a_k\}\) and announces the excluded index \(k\). The induced global measurement
    operator for guessing \(k\) is \(M_k = \sum_a A_a \otimes B^a_k\), and the error probability is

    \[
        \sum_{k} p_k \sum_a \operatorname{Tr}\!\left[(A_a \otimes B^a_k)\, \rho_k\right].
    \]

    Minimizing this jointly over \(\{A_a\}\) and \(\{B^a_k\}\) is bilinear, so it is solved by a
    see-saw (alternating optimization), exactly as in
    [`NonlocalGame.quantum_value_lower_bound`][toqito.nonlocal_games.nonlocal_game.NonlocalGame].
    Each half is a semidefinite program:

    With Alice's POVM \(\{A_a\}\) fixed, Bob's conditional ensemble is
    \(\sigma^a_k = \operatorname{Tr}_A[(A_a \otimes \mathbb{I}_B)\rho_k]\) and he solves, for every
    outcome \(a\),

    \[
        \begin{aligned}
            \text{minimize:} \quad & \sum_k p_k \operatorname{Tr}[B^a_k\, \sigma^a_k], \\
            \text{subject to:} \quad & \sum_k B^a_k = \mathbb{I}_B, \qquad B^a_k \succeq 0.
        \end{aligned}
    \]

    With Bob's POVMs fixed, write \(\tau_a = \sum_k p_k \operatorname{Tr}_B[(\mathbb{I}_A \otimes
    B^a_k)\rho_k]\); Alice solves

    \[
        \begin{aligned}
            \text{minimize:} \quad & \sum_a \operatorname{Tr}[A_a\, \tau_a], \\
            \text{subject to:} \quad & \sum_a A_a = \mathbb{I}_A, \qquad A_a \succeq 0.
        \end{aligned}
    \]

    Each returned value is the error of a concrete one-way LOCC strategy, hence an upper bound on
    the optimal one-way LOCC exclusion error and at least the global exclusion error
    (`global` \(\le\) `PPT` \(\le\) `LOCC`). Because the see-saw can converge to a local optimum,
    the procedure is restarted from `reps` random initial POVMs and the smallest value is returned.

    Args:
        states: Bipartite states, each given as a density matrix (or a vector, which is converted
            to a density matrix) on the \(\mathcal{H}_A \otimes \mathcal{H}_B\) system.
        probs: Respective selection probabilities. Defaults to the uniform distribution.
        dim: The two subsystem dimensions `[dim_A, dim_B]`. Their product must equal the dimension
            of each state.
        num_alice_outcomes: Number of measurement outcomes available to Alice. Defaults to the
            number of states. Allowing more outcomes can only tighten (lower) the value.
        reps: Number of random restarts of the see-saw. Defaults to 5.
        tol: Convergence tolerance on the error decrease between iterations.
        max_iters: Maximum number of see-saw iterations per restart.
        seed: Optional base seed for the random initial POVMs, for reproducibility. Restart `r`
            uses ``seed + r``.

    Returns:
        The smallest one-way LOCC exclusion error found across the random restarts.

    Examples:
        Three non-antidistinguishable two-qubit states whose separable (and hence LOCC) exclusion
        error is strictly larger than the global one.

        ```python exec="1" source="above" result="text"
        from toqito.state_opt import state_exclusion_locc
        import numpy as np

        vecs = [
            np.array([[0], [1], [1], [1]], dtype=complex),
            np.array([[0], [0], [1], [1]], dtype=complex),
            np.array([[1], [0], [1], [1]], dtype=complex),
        ]
        states = [v @ v.conj().T / float(np.linalg.norm(v) ** 2) for v in vecs]

        val = state_exclusion_locc(states, dim=[2, 2], reps=3, seed=1)
        print(f"One-way LOCC exclusion error: {np.around(val, decimals=2)}")
        ```

    """
    if not states:
        raise ValueError("At least one state must be provided.")
    if dim is None or len(dim) != 2:
        raise ValueError("Argument `dim` must be a list `[dim_A, dim_B]` of the two subsystem dimensions.")

    states = [to_density_matrix(state) for state in states]
    num_states = len(states)
    if probs is None:
        probs = [1 / num_states] * num_states
    if not np.isclose(sum(probs), 1):
        raise ValueError("Probabilities must sum to 1.")

    dim_a, dim_b = int(dim[0]), int(dim[1])
    if states[0].shape[0] != dim_a * dim_b:
        raise ValueError("The product of `dim` must equal the dimension of the states.")
    num_a = num_states if num_alice_outcomes is None else num_alice_outcomes

    best = float("inf")
    for rep in range(reps):
        rep_seed = None if seed is None else seed + rep
        povm = random_povm(dim_a, 1, num_a, seed=rep_seed)
        alice = [povm[:, :, 0, a] for a in range(num_a)]

        prev = float("inf")
        current = float("inf")
        for _ in range(max_iters):
            bob = _optimize_bob(states, probs, dim_a, dim_b, alice)
            alice = _optimize_alice(states, probs, dim_a, dim_b, bob, num_a)
            current = _locc_error(states, probs, alice, bob, num_a)
            if abs(prev - current) < tol:
                break
            prev = current

        best = min(best, current)

    return best


def _optimize_bob(states, probs, dim_a, dim_b, alice):
    """Fix Alice's POVM and optimize Bob's conditional exclusion POVMs."""
    num_a, num_states = len(alice), len(states)
    bob = {(a, k): cvxpy.Variable((dim_b, dim_b), hermitian=True) for a in range(num_a) for k in range(num_states)}

    constraints = []
    error = 0
    for a in range(num_a):
        conditional = [
            partial_trace(np.kron(alice[a], np.identity(dim_b)) @ states[k], [0], [dim_a, dim_b])
            for k in range(num_states)
        ]
        for k in range(num_states):
            constraints.append(bob[a, k] >> 0)
            error += probs[k] * cvxpy.trace(bob[a, k] @ conditional[k])
        constraints.append(sum(bob[a, k] for k in range(num_states)) == np.identity(dim_b))

    cvxpy.Problem(cvxpy.Minimize(cvxpy.real(error)), constraints).solve()
    return {key: np.array(var.value) for key, var in bob.items()}


def _optimize_alice(states, probs, dim_a, dim_b, bob, num_a):
    """Fix Bob's conditional POVMs and optimize Alice's POVM."""
    num_states = len(states)
    alice = [cvxpy.Variable((dim_a, dim_a), hermitian=True) for _ in range(num_a)]

    constraints = [a >> 0 for a in alice]
    constraints.append(sum(alice) == np.identity(dim_a))

    error = 0
    for a in range(num_a):
        tau = sum(
            probs[k] * partial_trace(np.kron(np.identity(dim_a), bob[a, k]) @ states[k], [1], [dim_a, dim_b])
            for k in range(num_states)
        )
        error += cvxpy.trace(alice[a] @ tau)

    cvxpy.Problem(cvxpy.Minimize(cvxpy.real(error)), constraints).solve()
    return [np.array(a.value) for a in alice]


def _locc_error(states, probs, alice, bob, num_a):
    """Evaluate the exclusion error of a concrete one-way LOCC strategy."""
    num_states = len(states)
    total = 0.0
    for k in range(num_states):
        guess_k = sum(np.kron(alice[a], bob[a, k]) for a in range(num_a))
        total += probs[k] * np.real(np.trace(guess_k @ states[k]))
    return float(total)
