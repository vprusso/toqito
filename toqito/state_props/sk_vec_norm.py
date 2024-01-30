"""Compute the S(k)-norm of a vector."""


import numpy as np

from toqito.state_ops import schmidt_decomposition


def sk_vector_norm(rho: np.ndarray, k: int = 1, dim: int | list[int] = None) -> float:
    r"""Compute the S(k)-norm of a vector :cite:`Johnston_2010_AFamily`.

    The :math:`S(k)`-norm of of a vector :math:`|v \rangle` is
    defined as:

    .. math::
        \big|\big| |v\rangle \big|\big|_{s(k)} := \text{sup}_{|w\rangle} \Big\{
            |\langle w | v \rangle| : \text{Schmidt-rank}(|w\rangle) \leq k
        \Big\}

    It's also equal to the Euclidean norm of the vector of :math:`|v\rangle`'s
    k largest Schmidt coefficients.

    This function was adapted from QETLAB.

    Examples
    ========

    The smallest possible value of the :math:`S(k)`-norm of a pure state is
    :math:`\sqrt{\frac{k}{n}}`, and is attained exactly by the "maximally entangled
    states".

    >>> from toqito.states import max_entangled
    >>> from toqito.state_props import sk_vector_norm
    >>> import numpy as np
    >>>
    >>> # Maximally entagled state.
    >>> v = max_entangled(4)
    >>> sk_vector_norm(v)
    0.5

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param rho: A vector.
    :param k: An int.
    :param dim: The dimension of the two sub-systems. By default it's
                assumed to be equal.
    :return: The S(k)-norm of :code:`rho`.

    """
    dim_xy = rho.shape[0]

    # Set default dimension if none was provided.
    if dim is None:
        dim = int(np.round(np.sqrt(dim_xy)))

    # Allow the user to enter in a single integer for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim_xy / dim])  # pylint: disable=redefined-variable-type
        dim[1] = int(np.round(dim[1]))

    # It's faster to just compute the norm of `rho` directly if that will give
    # the correct answer.
    if k >= min(dim):
        nrm = np.linalg.norm(rho, 2)
    else:
        coef, _, _ = schmidt_decomposition(rho, dim, k)
        nrm = np.linalg.norm(coef)

    return nrm
