"""phase damping channel."""

import warnings

import numpy as np

from toqito.channel_ops.kraus_to_choi import kraus_to_choi


def phase_damping(
    input_mat: np.ndarray | None = None,
    gamma: float = 0,
    return_kraus_ops: bool = False,
) -> np.ndarray | list[np.ndarray]:
    r"""Return the phase damping channel [@nielsen2011quantum].

    The phase damping channel describes how quantum information is lost due to environmental interactions,
    causing dephasing in the computational basis without losing energy.

    The Kraus operators for the phase damping channel are:

    \[
        K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \\
        K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix},
    \]

    Args:
        input_mat: Deprecated. Passing a matrix here applies the channel to that matrix; this
            convenience path will be removed in a future release. Prefer
            `apply_channel(input_mat, phase_damping(gamma=...))`.
        gamma: The dephasing rate (between 0 and 1), representing the probability of phase decoherence.
        return_kraus_ops: If `True`, return the list of Kraus operators instead of the Choi matrix.

    Returns:
        The Choi matrix of the channel, or its list of Kraus operators when `return_kraus_ops` is
        `True`. When the deprecated `input_mat` argument is provided, the channel applied to that
        input is returned instead.

    Examples:
        Apply the channel (returned as a Choi matrix) to a state via `apply_channel`:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels.phase_damping import phase_damping
        from toqito.channel_ops import apply_channel

        rho = np.array([[1, 0.5], [0.5, 1]])
        result = apply_channel(rho, phase_damping(gamma=0.2))

        print(result)
        ```

    """
    if not (0 <= gamma <= 1):
        raise ValueError("Gamma must be between 0 and 1.")

    k0 = np.diag([1, np.sqrt(1 - gamma)])
    k1 = np.diag([0, np.sqrt(gamma)])

    if input_mat is None:
        kraus_ops = [k0, k1]
        return kraus_ops if return_kraus_ops else kraus_to_choi(kraus_ops)

    if input_mat.shape != (2, 2):
        raise ValueError("Input matrix must be 2x2 for the phase damping channel.")

    warnings.warn(
        "Passing `input_mat` to `phase_damping` is deprecated; "
        "use `apply_channel(input_mat, phase_damping(...))` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    input_mat = np.asarray(input_mat, dtype=complex)

    return k0 @ input_mat @ k0.conj().T + k1 @ input_mat @ k1.conj().T
