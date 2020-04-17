"""Computes the von Neumann or Rényi entropy of a density matrix."""
import warnings
import numpy as np
from numpy import linalg as lin_alg


def entropy(rho: np.ndarray, log_base: int = 2, alpha: float = 1) -> float:
    r"""
    Compute the von Neumann or Renyi entropy of a density matrix [1]_.

    Calculates the entropy of `rho`, computed with logarithms in the base
    specified by `log_base`. If `alpha = 1`, then this is the von Neumann
    entropy. If `alpha > 1`, then this is the Renyi-`alpha` entropy.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of $u$ may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    Calculating the entropy of :math:`\rho` in `toqito` can be done as follows.

    >>> from toqito.states.distance.entropy import entropy
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    >>> )
    >>> entropy(test_input_mat)
    5.88418203051333e-15

    It is also possible to tweak the `log` parameter of the `entropy` function.
    For instance, we may compute the entropy of :math:`\rho` with the parameter
    `log=10` in `toqito` as follows.

    >>> from toqito.states.distance.entropy import entropy
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    >>> )
    >>> entropy(test_input_mat, 10)
    1.7713152911315034e-15

    References
    ==========
    .. [1] Wikipedia: Von Neumann entropy
        https://en.wikipedia.org/wiki/Von_Neumann_entropy

    :param rho: A density matrix.
    :param log_base: Log base. Default set to 2.
    :param alpha: Default parameter set to 1.
    """
    eigs, _ = lin_alg.eig(rho)
    eigs = [eig for eig in eigs if eig > 0]

    # If `alpha == 1`, compute the von Neumann entropy.
    if np.abs(alpha - 1) <= np.finfo(float).eps ** (3 / 4):
        if log_base == 2:
            ent = -np.sum(np.real(eigs * np.log2(eigs)))
        else:
            ent = -np.sum(np.real(eigs * np.log(eigs))) / np.log(log_base)
        return ent

    if alpha >= 0:

        # Renyi-alpha entropy with `alpha < float("inf")`
        if alpha < float("inf"):
            ent = np.log(np.sum(eigs ** alpha)) / (np.log(log_base) * (1 - alpha))

            # Check whether or not we ran into numerical problems due to
            # `alpha` being large. If so, compute the infinity-entropy instead.
            if ent == float("inf"):
                alpha = float("inf")
                warnings.warn(
                    "LargeAlpha: Numerical problems were encountered "
                    "due to a large value of `alpha`. Computing the "
                    "entropy with `alpha = float('inf')` instead."
                )

        # Do not merge the following if statement with the previous one: we
        # need them separate, since this one catches a warning from the
        # previous block.
        if alpha == float("inf"):
            # Renyi-infinity entropy.
            ent = -np.log(np.max(eigs)) / np.log(log_base)

        return ent

    raise ValueError("InvalidAlpha: The `alpha` parameter must be " "non-negative.")
