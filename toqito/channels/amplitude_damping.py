"""Generates the (generalized) amplitude damping channel."""

import warnings

import numpy as np


def amplitude_damping(
    input_mat: np.ndarray | None = None,
    gamma: float = 0,
    prob: float = 1,
) -> np.ndarray | list[np.ndarray]:
    r"""Return the Kraus operators of the generalized amplitude damping channel.

    The generalized amplitude damping channel models energy dissipation in a quantum system,
    where the system can lose energy to its environment with a certain probability. The
    channel is defined by two parameters: `gamma` (the damping rate) and `prob` (the
    probability of energy loss).

    To also include standard implementation of amplitude damping, we have set `prob = 1` as the default implementation.

    !!! note
        This channel is defined for qubit systems in the standard literature [@khatri2020information].


    The Kraus operators for the generalized amplitude damping channel are given by:

    \[
        K_0 = \sqrt{p} \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \\
        K_1 = \sqrt{p}  \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}, \\
        K_2 = \sqrt{1 - p} \begin{pmatrix} \sqrt{1 - \gamma} & 0 \\ 0 & 1 \end{pmatrix}, \\
        K_3 = \sqrt{1 - p}  \begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}, \\
    \]

    These operators describe the evolution of a quantum state under the generalized amplitude
    damping process.

    Args:
        input_mat: Deprecated. Passing a matrix here applies the channel to that matrix; this
            convenience path will be removed in a future release. Prefer
            `apply_channel(amplitude_damping(gamma=..., prob=...), input_mat)`.
        gamma: The damping rate, a float between 0 and 1. Represents the probability of energy dissipation.
        prob: The probability of energy loss, a float between 0 and 1.

    Returns:
        The list of Kraus operators describing the channel. When the deprecated `input_mat`
        argument is provided, the channel applied to that input is returned instead.

    Examples:
        Obtain the Kraus operators and apply the channel via `apply_channel`:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import amplitude_damping
        from toqito.channel_ops import apply_channel

        rho = np.array([[1, 0], [0, 0]])  # |0><0|
        result = apply_channel(rho, amplitude_damping(gamma=0.1, prob=0.5))

        print(result)
        ```

    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    if not (0 <= gamma <= 1):
        raise ValueError("Gamma (damping rate) must be between 0 and 1.")

    k0 = np.sqrt(prob) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = np.sqrt(prob) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
    k2 = np.sqrt(1 - prob) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]])
    k3 = np.sqrt(1 - prob) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])

    if input_mat is None:
        return [k0, k1, k2, k3]

    if input_mat.shape != (2, 2):
        raise ValueError("Input matrix must be 2x2 for the generalized amplitude damping channel.")

    warnings.warn(
        "Passing `input_mat` to `amplitude_damping` is deprecated; "
        "use `apply_channel(amplitude_damping(...), input_mat)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    input_mat = np.asarray(input_mat, dtype=complex)

    return (
        k0 @ input_mat @ k0.conj().T
        + k1 @ input_mat @ k1.conj().T
        + k2 @ input_mat @ k2.conj().T
        + k3 @ input_mat @ k3.conj().T
    )
