"""Bures angle, also known as Bures arc, Bures length or quantum angle is a distance metric.

The Bures angle metric is a measure of the statistical distance between quantum states.
"""

import numpy as np

from toqito.state_metrics import fidelity


def bures_angle(rho_1: np.ndarray, rho_2: np.ndarray, decimals: int = 10) -> float:
    r"""Compute the Bures angle of two density matrices [@WikiBures].

    Calculate the Bures angle between two density matrices `rho_1` and `rho_2` defined by:

    \[
        \arccos{\sqrt{F (\rho_1, \rho_2)}}
    \]

    where \(F(\cdot)\) denotes the fidelity between \(\rho_1\) and \(\rho_2\). The return is a value between
    \(0\) and \(\pi / 2\), with \(0\) corresponding to matrices `rho_1 = rho_2` and \(\pi / 2\)
    corresponding to the case `rho_1` and `rho_2` with orthogonal support.

    Examples:

    Consider the following Bell state

    \[
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.
    \]

    The corresponding density matrix of \(u\) may be calculated by:

    \[
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).
    \]

    In the event where we calculate the Bures angle between states that are identical, we should obtain the value of
    \(0\). This can be observed in `|toqito⟩` as follows.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.state_metrics import bures_angle
    
    rho = 1 / 2 * np.array(
        [[1, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]
    )
    sigma = rho
    
    print(bures_angle(rho, sigma))
    ```

    Raises:
        ValueError: If matrices are not of equal dimension.

    Args:
        rho_1: Density operator.
        rho_2: Density operator.
        decimals: Number of decimal places to round to (default 10).

    Returns:
        The Bures angle between `rho_1` and `rho_2`.

    """
    # Perform error checking.
    if not np.all(rho_1.shape == rho_2.shape):
        raise ValueError("InvalidDim: `rho_1` and `rho_2` must be matrices of the same size.")
    # Round fidelity to only 10 decimals to avoid error when `rho_1 = rho_2`.
    return np.real(np.arccos(np.sqrt(np.round(fidelity(rho_1, rho_2), decimals))))
