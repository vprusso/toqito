"""Schmidt number 3 bound entangled grid state on two five-level systems."""

import numpy as np

from toqito.states.generalized_grid_state import generalized_grid_state


def krebs_gachechiladze() -> np.ndarray:
    r"""Produce the Schmidt number 3 PPT state of Krebs and Gachechiladze on two ququints [@krebs2024high].

    This is the generalized grid state (`toqito.states.generalized_grid_state`) on the \(5 \times 5\) grid
    with the 13 hyperedges of [@krebs2024high]: five loops at \((0,0)\), \((1,0)\), \((0,1)\), \((4,1)\) and
    \((1,4)\); the loops at \((3,2)\) and \((2,3)\), each listed twice; the edges \(\{(1,2), (3,4)\}\),
    \(\{(2,1), (4,3)\}\) and \(\{(2,2), (3,3)\}\); and the hyperedge \(\{(0,2), (1,1), (2,0)\}\). It is the
    smallest known PPT state with Schmidt number 3, which had been believed to require local dimension 6.
    The state has rank 11, is PPT, and the realignment criterion certifies it is entangled.

    Returns:
        The 25-by-25 state.

    Examples:
        The state is PPT and has rank 11.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import krebs_gachechiladze
        from toqito.state_props import is_ppt
        rho = krebs_gachechiladze()
        print(is_ppt(rho, dim=[5, 5]), np.linalg.matrix_rank(rho))
        ```

    """
    hyperedges = [
        [(0, 0)],
        [(1, 0)],
        [(0, 1)],
        [(4, 1)],
        [(1, 4)],
        [(3, 2)],
        [(2, 3)],
        [(3, 2)],
        [(2, 3)],
        [(1, 2), (3, 4)],
        [(2, 1), (4, 3)],
        [(2, 2), (3, 3)],
        [(0, 2), (1, 1), (2, 0)],
    ]
    return generalized_grid_state([5, 5], hyperedges)
