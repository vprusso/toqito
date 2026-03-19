"""Generates a random quantum channel (superoperator)."""

import numpy as np

from toqito.matrix_ops import partial_trace
from toqito.rand import random_density_matrix


def random_superoperator(
    dim: int | list[int],
    is_real: bool = False,
    is_trace_preserving: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate a random quantum channel as a Choi matrix.

    Generates the Choi matrix of a random completely positive (CP) map. By default,
    the channel is also trace-preserving (CPTP), making it a valid quantum channel.

    The method generates a random density matrix of dimension \(d_{\text{in}} \cdot d_{\text{out}}\)
    (distributed according to the Haar/Hilbert-Schmidt measure) and uses it as the Choi matrix.
    For trace-preserving channels, the Choi matrix is adjusted so that its partial trace over
    the output system equals the identity:

    \[
        \text{Tr}_{\text{out}}(J(\Phi)) = I_{d_{\text{in}}}
    \]

    This approach follows the method used in QETLAB's `RandomSuperoperator`.

    Args:
        dim: The dimension of the channel. If an integer, the channel maps
            \(d \times d\) matrices to \(d \times d\) matrices. If a list
            `[dim_in, dim_out]`, the channel maps `dim_in`-by-`dim_in` matrices
            to `dim_out`-by-`dim_out` matrices.
        is_real: If `True`, the Choi matrix has only real entries. Default `False`.
        is_trace_preserving: If `True` (default), the channel is trace-preserving
            (i.e., a valid quantum channel / CPTP map). If `False`, the channel is
            only completely positive (CP).
        seed: A seed for numpy's random number generator.

    Returns:
        The Choi matrix of the random channel, of shape
        `(dim_in * dim_out, dim_in * dim_out)`.

    Raises:
        ValueError: If dimensions are not positive integers.

    Examples:
        Generate a random quantum channel on qubits.

        ```python exec="1" source="above" result="text"
        from toqito.rand import random_superoperator
        from toqito.channel_props import is_quantum_channel

        choi = random_superoperator(2, seed=42)
        print(f"Shape: {choi.shape}")
        print(f"Is valid quantum channel: {is_quantum_channel(choi)}")
        ```

        Generate a random channel from a 2-dimensional to a 3-dimensional system.

        ```python exec="1" source="above" result="text"
        from toqito.rand import random_superoperator

        choi = random_superoperator([2, 3], seed=7)
        print(f"Shape: {choi.shape}")
        ```

    """
    if isinstance(dim, int):
        dim_in, dim_out = dim, dim
    elif isinstance(dim, list) and len(dim) == 2:
        dim_in, dim_out = dim
    else:
        raise ValueError("`dim` must be a positive integer or a list of two positive integers.")

    if dim_in < 1 or dim_out < 1:
        raise ValueError("Dimensions must be positive integers.")

    prod_dim = dim_in * dim_out

    # Generate a random density matrix as the Choi matrix.
    choi = random_density_matrix(prod_dim, is_real=is_real, seed=seed)

    if is_trace_preserving:
        # Adjust so that Tr_out(Choi) = I_in / dim_in.
        # The Choi matrix of a TP map satisfies Tr_out(J) = I_in.
        # We have a random density matrix (trace 1). We need to rescale
        # and correct it: J = dim_in * (A^{-1/2} ⊗ I_out) choi (A^{-1/2} ⊗ I_out)
        # where A = Tr_out(choi).
        marginal = partial_trace(choi, sys=[1], dim=[dim_in, dim_out])

        # Compute A^{-1/2} via eigendecomposition.
        eigvals, eigvecs = np.linalg.eigh(marginal)
        # Clamp small eigenvalues to avoid numerical issues.
        eigvals = np.maximum(eigvals, 1e-15)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T

        correction = np.kron(inv_sqrt, np.eye(dim_out))
        choi = correction @ choi @ correction.conj().T

        # Ensure Hermiticity after numerical operations.
        choi = (choi + choi.conj().T) / 2

    return choi
