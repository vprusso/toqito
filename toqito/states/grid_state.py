"""Quantum grid states built from the edges of a graph on a product basis."""

import numpy as np


def grid_state(dims: list[int], edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    r"""Produce a quantum grid state [@lockhart2018entanglement].

    The vertices of an \(m \times n\) grid are identified with the product basis \(|i, j\rangle\) of
    \(\mathbb{C}^m \otimes \mathbb{C}^n\). An edge joining the vertices \((i, j)\) and \((k, l)\) carries the
    pure state

    \[
        |e\rangle = \frac{1}{\sqrt{2}} \left( |i, j\rangle - |k, l\rangle \right),
    \]

    and the grid state of an edge set \(E\) is the uniform mixture

    \[
        \rho = \frac{1}{|E|} \sum_{e \in E} |e\rangle\langle e|.
    \]

    Whether the result is separable, has negative partial transpose, or is bound entangled is a property of
    the graph. The cross-hatch graph on the \(3 \times 3\) grid (`toqito.states.cross_hatch`) gives a bound
    entangled state, while a path on the \(2 \times 3\) grid gives a PPT state of total dimension 6, which
    is separable.

    Args:
        dims: The grid dimensions \([m, n]\).
        edges: The edges, each a pair of vertices \(((i, j), (k, l))\) with \(0 \leq i, k < m\) and
            \(0 \leq j, l < n\).

    Returns:
        The grid state on \(\mathbb{C}^m \otimes \mathbb{C}^n\).

    Raises:
        ValueError: If no edges are given, or a vertex lies outside the grid.

    Examples:
        A path through the \(2 \times 3\) grid gives a separable state, since every PPT state of total
        dimension 6 is separable.

        ```python exec="1" source="above" result="text"
        from toqito.states import grid_state
        from toqito.state_props import is_ppt
        rho = grid_state([2, 3], [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
        print(is_ppt(rho, dim=[2, 3]))
        ```

    """
    if not edges:
        raise ValueError("At least one edge is required.")
    dim_m, dim_n = dims
    rho = np.zeros((dim_m * dim_n, dim_m * dim_n))
    for first, second in edges:
        for row, col in (first, second):
            if not (0 <= row < dim_m and 0 <= col < dim_n):
                raise ValueError(f"Vertex ({row}, {col}) lies outside the {dim_m} x {dim_n} grid.")
        vec = np.zeros((dim_m * dim_n, 1))
        vec[first[0] * dim_n + first[1]] = 1 / np.sqrt(2)
        vec[second[0] * dim_n + second[1]] = -1 / np.sqrt(2)
        rho += vec @ vec.T
    return rho / len(edges)
