"""Quasi-diagonal-symmetric states."""

from math import comb

import numpy as np

from toqito.states.dicke import dicke


def quasi_ds(num_qubits: int, z_param: float, sigma: int) -> np.ndarray:
    r"""Produce a quasi-diagonal-symmetric state [@tura2018separability].

    A one-parameter family of permutationally symmetric states on an odd number \(n = 2K + 1\) of qubits.
    Each member is a diagonal symmetric (DS) state in the Dicke basis \(|D^n_k\rangle\), \(k = 0, \dots, n\),
    plus a coherence between the two GHZ corners, hence "quasi-DS". Let \(f_k(Z)\) be defined for every
    integer \(k\) by \(f_0 = 1\), \(f_1 = 1 + Z\) and the recurrence
    \(f_{k+2}(Z) = (2 + Z) f_{k+1}(Z) - f_k(Z)\), which gives \(f_{-k}(Z) = f_{k-1}(Z)\). With
    \(Z > 0\) and \(\sigma = \pm 1\), the state in the Dicke basis is

    \[
        \rho(Z) = \frac{1}{2 (4 + Z)^K} \left( \sum_{k=0}^{n} \binom{n}{k} f_{K-k}(Z) \, |D^n_k\rangle\langle D^n_k|
            + \sigma \big( |D^n_0\rangle\langle D^n_n| + |D^n_n\rangle\langle D^n_0| \big) \right),
    \]

    and it is returned in the computational basis through the isometry \(V = \sum_k |D^n_k\rangle\langle k|\).

    The state is PPT across every bipartition of the qubits. For \(K > 1\) it is extreme in the PPT set
    and therefore not fully separable, which is a statement about the \(n\)-party partition: every
    bipartite cut of the state is separable, so no bipartite test detects its entanglement.

    Args:
        num_qubits: The number of qubits \(n\), an odd integer at least 5.
        z_param: The parameter \(Z > 0\).
        sigma: The sign of the GHZ coherence, either `1` or `-1`.

    Returns:
        The quasi-DS state as a \(2^n \times 2^n\) matrix.

    Raises:
        ValueError: If `num_qubits` is even or smaller than 5, if `z_param` is not positive, or if `sigma`
            is not `1` or `-1`.

    Examples:
        The five-qubit state is PPT across the 1|4 and the 2|3 cuts.

        ```python exec="1" source="above" result="text"
        from toqito.states import quasi_ds
        from toqito.state_props import is_ppt
        rho = quasi_ds(5, 1.0, 1)
        print(is_ppt(rho, dim=[2, 16]), is_ppt(rho, dim=[4, 8]))
        ```

    """
    if num_qubits < 5 or num_qubits % 2 == 0:
        raise ValueError("The number of qubits must be an odd integer at least 5.")
    if z_param <= 0:
        raise ValueError("The parameter z_param must be positive.")
    if sigma not in (1, -1):
        raise ValueError("The sign sigma must be 1 or -1.")

    half = num_qubits // 2
    # f_k for k = 0, ..., half; negative indices follow from f_{-k} = f_{k-1}.
    f_vals = [1.0, 1.0 + z_param]
    for _ in range(2, half + 1):
        f_vals.append((2 + z_param) * f_vals[-1] - f_vals[-2])

    def f_of(k_idx: int) -> float:
        return f_vals[k_idx] if k_idx >= 0 else f_vals[-k_idx - 1]

    rho_dicke = np.diag([comb(num_qubits, k) * f_of(half - k) for k in range(num_qubits + 1)]).astype(float)
    rho_dicke[0, num_qubits] = rho_dicke[num_qubits, 0] = sigma
    rho_dicke /= 2 * (4 + z_param) ** half

    isometry = np.column_stack([dicke(num_qubits, k) for k in range(num_qubits + 1)])
    return isometry @ rho_dicke @ isometry.T
