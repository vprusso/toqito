"""Compute the S(k)-norm of a vector."""

import numpy as np

from toqito.state_ops import schmidt_decomposition


def sk_vector_norm(rho: np.ndarray, k: int = 1, dim: int | list[int] | None = None) -> float | np.floating:
    r"""Compute the S(k)-norm of a vector [@Johnston_2010_AFamily].

    The \(S(k)\)-norm of of a vector \(|v \rangle\) is
    defined as:

    \[
        \big|\big| |v\rangle \big|\big|_{s(k)} := \text{sup}_{|w\rangle} \Big\{
            |\langle w | v \rangle| : \text{Schmidt-rank}(|w\rangle) \leq k
        \Big\}
    \]

    It's also equal to the Euclidean norm of the vector of \(|v\rangle\)'s
    k largest Schmidt coefficients.

    This function was adapted from QETLAB.

    Examples:
    The smallest possible value of the \(S(k)\)-norm of a pure state is
    \(\sqrt{\frac{k}{n}}\), and is attained exactly by the "maximally entangled
    states".

    ```python exec="1" source="above"
    from toqito.states import max_entangled
    from toqito.state_props import sk_vector_norm
    import numpy as np
    # Maximally entagled state.
    v = max_entangled(4)
    print(sk_vector_norm(v))
    ```

    Args:
        rho: A vector.
        k: An int.
        dim: The dimension of the two sub-systems. By default it's assumed to be equal.

    Returns:
        The S(k)-norm of `rho`.

    """
    dim_xy = rho.shape[0]

    # Set default dimension if none was provided.
    if dim is None:
        dim_val = int(np.round(np.sqrt(dim_xy)))
    elif isinstance(dim, int):
        dim_val = dim
    else:
        dim_val = None

    # Allow the user to enter in a single integer for dimension.
    if dim_val is not None:
        dim_arr = np.array([dim_val, dim_xy / dim_val])
        dim_arr[1] = int(np.round(dim_arr[1]))
    else:
        dim_arr = np.array(dim)

    # It's faster to just compute the norm of `rho` directly if that will give
    # the correct answer.
    if k >= min(dim_arr):
        nrm = np.linalg.norm(rho, 2)
    else:
        coef, _, _ = schmidt_decomposition(rho, dim_arr, k)
        nrm = np.linalg.norm(coef)

    return nrm
