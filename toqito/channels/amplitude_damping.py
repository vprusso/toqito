"""Generates the (generalized) amplitude damping channel."""

import numpy as np

from toqito.channel_ops.kraus_to_choi import kraus_to_choi


def amplitude_damping(
    gamma: float = 0,
    prob: float = 1,
    return_kraus_ops: bool = False,
) -> np.ndarray | list[np.ndarray]:
    r"""Return the generalized amplitude damping channel.

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
        gamma: The damping rate, a float between 0 and 1. Represents the probability of energy dissipation.
        prob: The probability of energy loss, a float between 0 and 1.
        return_kraus_ops: If `True`, return the list of Kraus operators instead of the Choi matrix.

    Returns:
        The Choi matrix of the channel, or its list of Kraus operators when `return_kraus_ops` is
        `True`.

    Examples:
        Apply the channel (returned as a Choi matrix) to a state via `apply_channel`:

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

    kraus_ops = [k0, k1, k2, k3]
    return kraus_ops if return_kraus_ops else kraus_to_choi(kraus_ops)
