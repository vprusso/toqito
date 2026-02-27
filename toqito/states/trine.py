"""Trine states are states of linear polarization separated by 60°."""

import numpy as np

from toqito.states import basis


def trine() -> list[np.ndarray]:
    r"""Produce the set of trine states (Slide 6 of [@Yard_2017_Lecture11]).

    The trine states are formally defined as:

    \[
        u_1 = |0\rangle, \quad
        u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
        u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).
    \]

    Examples:
        Generating the trine states can be done by simply invoking the function:

        ```python exec="1" source="above"
        from toqito.states import trine
        print(trine())
        ```

    Returns:
        Vector of trine states.

    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    return [
        e_0,
        -1 / 2 * (e_0 + np.sqrt(3) * e_1),
        -1 / 2 * (e_0 - np.sqrt(3) * e_1),
    ]
