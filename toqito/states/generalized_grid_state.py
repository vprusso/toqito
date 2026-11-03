"""Generalized grid states built from hyperedges on a product basis."""

import numpy as np


def generalized_grid_state(dims: list[int], hyperedges: list[list[tuple[int, int]]]) -> np.ndarray:
    r"""Produce a generalized grid state [@krebs2024high].

    The generalized grid state extends the grid state of [@lockhart2018entanglement]
    (`toqito.states.grid_state`) by letting a hyperedge span any number of vertices of the \(m \times n\)
    grid, whose vertices are identified with the product basis \(|i, j\rangle\) of
    \(\mathbb{C}^m \otimes \mathbb{C}^n\). A hyperedge \(h\) carries the unnormalized vector

    \[
        |h\rangle = \sum_{(i, j) \in h} |i, j\rangle,
    \]

    with no relative sign, and the state is the trace-normalized sum over the multiset \(H\) of hyperedges,

    \[
        \rho = \frac{\sum_{h \in H} |h\rangle\langle h|}{\operatorname{Tr} \sum_{h \in H} |h\rangle\langle h|}.
    \]

    One-vertex hyperedges are loops, and the same hyperedge may be listed more than once to give it more
    weight. Hyperedges with three or more vertices are what allow high Schmidt numbers. Most small hyperedge
    sets give a state with negative partial transpose; the \(5 \times 5\) example of
    `toqito.states.krebs_gachechiladze` is PPT with Schmidt number 3.

    Args:
        dims: The grid dimensions \([m, n]\).
        hyperedges: The hyperedges, each a list of vertices \((i, j)\) with \(0 \leq i < m\) and
            \(0 \leq j < n\).

    Returns:
        The generalized grid state on \(\mathbb{C}^m \otimes \mathbb{C}^n\).

    Raises:
        ValueError: If no hyperedges are given, a hyperedge is empty, or a vertex lies outside the grid.

    Examples:
        A single two-vertex hyperedge on the \(2 \times 2\) grid is a Bell state, which has negative partial
        transpose.

        ```python exec="1" source="above" result="text"
        from toqito.states import generalized_grid_state
        from toqito.state_props import is_ppt
        rho = generalized_grid_state([2, 2], [[(0, 0), (1, 1)]])
        print(is_ppt(rho, dim=[2, 2]))
        ```

    """
    if not hyperedges:
        raise ValueError("At least one hyperedge is required.")
    dim_m, dim_n = dims
    rho = np.zeros((dim_m * dim_n, dim_m * dim_n))
    for hyperedge in hyperedges:
        if not hyperedge:
            raise ValueError("Hyperedges must contain at least one vertex.")
        vec = np.zeros((dim_m * dim_n, 1))
        for row, col in hyperedge:
            if not (0 <= row < dim_m and 0 <= col < dim_n):
                raise ValueError(f"Vertex ({row}, {col}) lies outside the {dim_m} x {dim_n} grid.")
            vec[row * dim_n + col] += 1
        rho += vec @ vec.T
    return rho / np.trace(rho)
