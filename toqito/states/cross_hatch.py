"""Cross-hatch bound entangled grid state on two qutrits."""

import numpy as np

from toqito.states.grid_state import grid_state


def cross_hatch() -> np.ndarray:
    r"""Produce the cross-hatch grid state [@lockhart2018entanglement].

    The cross-hatch state is the grid state (`toqito.states.grid_state`) on the \(3 \times 3\) grid with
    the four edges

    \[
        \{(0, 0), (1, 2)\}, \quad \{(1, 0), (2, 2)\}, \quad \{(0, 1), (2, 0)\}, \quad \{(0, 2), (2, 1)\},
    \]

    as listed in [@imai2021bound]. Every vertex is used at most once, so the four edge states are
    orthogonal and the state has rank 4. It is PPT, and the realignment (CCNR) criterion certifies that it
    is entangled, so it is bound entangled.

    Returns:
        The 9-by-9 cross-hatch state.

    Examples:
        The state is PPT yet detected as entangled by the realignment criterion.

        ```python exec="1" source="above" result="text"
        from toqito.states import cross_hatch
        from toqito.state_props import is_ppt, is_separable
        rho = cross_hatch()
        print(is_ppt(rho, dim=[3, 3]), is_separable(rho, dim=[3, 3]))
        ```

    """
    return grid_state([3, 3], [((0, 0), (1, 2)), ((1, 0), (2, 2)), ((0, 1), (2, 0)), ((0, 2), (2, 1))])
