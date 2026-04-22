"""Implements the bitflip quantum gate channel."""

import warnings

import numpy as np


def bitflip(
    input_mat: np.ndarray | None = None,
    prob: float = 0,
) -> np.ndarray | list[np.ndarray]:
    r"""Return the Kraus operators of the bitflip channel.

    The *bitflip channel* is a quantum channel that flips a qubit from \(|0\rangle\) to \(|1\rangle\)
    and from \(|1\rangle\) to \(|0\rangle\) with probability \(p\).
    It is defined by the following operation:

    \[
        \mathcal{E}(\rho) = (1-p) \rho + p X \rho X
    \]

    where \(X\) is the Pauli-X (NOT) gate given by:

    \[
        X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
    \]

    The Kraus operators for this channel are:

    \[
        K_0 = \sqrt{1-p} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
        K_1 = \sqrt{p} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
    \]

    Args:
        input_mat: Deprecated. Passing a matrix here applies the channel to that matrix; this
            convenience path will be removed in a future release. Prefer
            `apply_channel(bitflip(prob=...), input_mat)`.
        prob: The probability of a bitflip occurring.

    Returns:
        The list of Kraus operators describing the channel. When the deprecated `input_mat`
        argument is provided, the channel applied to that input is returned instead.

    Examples:
        Obtain the Kraus operators for the bitflip channel with probability 0.3:

        ```python exec="1" source="above" result="text"
        from toqito.channels import bitflip

        print(bitflip(prob=0.3))
        ```


        Apply the bitflip channel to the state \(|0\rangle\) via `apply_channel`:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import bitflip
        from toqito.channel_ops import apply_channel

        rho = np.array([[1, 0], [0, 0]])  # |0><0|
        print(apply_channel(rho, bitflip(prob=0.3)))
        ```

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    k0 = np.sqrt(1 - prob) * np.eye(2)
    k1 = np.sqrt(prob) * np.array([[0, 1], [1, 0]])

    if input_mat is None:
        return [k0, k1]

    if input_mat.shape != (2, 2):
        raise ValueError("Input matrix must be 2x2 for the bitflip channel.")

    warnings.warn(
        "Passing `input_mat` to `bitflip` is deprecated; "
        "use `apply_channel(bitflip(...), input_mat)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    input_mat = np.asarray(input_mat, dtype=complex)

    return k0 @ input_mat @ k0.conj().T + k1 @ input_mat @ k1.conj().T
