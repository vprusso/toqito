"""Calculates the Rényi entropy metric of a quantum state."""

import numpy as np

from toqito.matrix_props import is_density
from toqito.state_props import von_neumann_entropy


def renyi_entropy(rho: np.ndarray, alpha: float) -> float:
    r"""Compute the Rényi entropy of a density matrix :footcite:`Muller_2013_Renyi_Generalization`.

    Let :math:`P \in \text{Pos}(\mathcal{X})` be a positive semidefinite operator, for a complex
    Euclidean space :math:`\mathcal{X}`. Then one defines the *Rényi entropy of order*
    :math:`\alpha\geqslant0` as

    .. math::
        H_{\alpha}(P) = H_{\alpha}(\lambda(P)),

    where :math:`\lambda(P)` is the vector of eigenvalues of :math:`P` and where the function
    :math:`H(\cdot)` is the classical Rényi entropy of order :math:`\alpha` defined as

    .. math::
        H_{\alpha}(u) = \frac{1}{1-\alpha}\log\left(\sum_{a \in \Sigma} u(a)^{\alpha}\right),

    where the :math:`\log` function is assumed to be the base-2 logarithm, and where
    :math:`\Sigma` is an alphabet where :math:`u \in [0, \infty)^{\Sigma}` is a vector of
    nonnegative real numbers indexed by :math:`\Sigma`. It recovers the von Neumann entropy for
    :math:`\alpha=1` and the min-entropy for :math:`\alpha=+\infty`.

    Examples
    ==========

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}} \left(|00 \rangle + |11 \rangle \right) \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    Calculating the Rényi entropy of order :math:`2` of :math:`\rho` in :code:`|toqito⟩` can be
    done as follows.

    .. jupyter-execute::

        from toqito.state_props import renyi_entropy
        import numpy as np
        test_input_mat = np.array(
                [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0],
                [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
            )
        renyi_entropy(test_input_mat, 2)

    Consider the density operator corresponding to the maximally mixed state of dimension two

    .. math::
        \rho = \frac{1}{2}
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}.

    As this state is maximally mixed, the Rényi entropy of :math:`\rho` is
    equal to one for all orders :math:`\alpha`. We can see this in :code:`|toqito⟩` as follows.

    .. jupyter-execute::

        from toqito.state_props import renyi_entropy
        import numpy as np
        rho = 1/2 * np.identity(2)
        renyi_entropy(rho, 3/2)

    References
    ==========
    .. footbibliography::



    :param rho: Density operator.
    :param alpha: Order for the Rényi entropy. Note that numerical instability may happen for small
        positive values because of the computation of the spectral decomposition.
    :return: The Rényi entropy of order :code:`alpha` of :code:`rho`.

    """
    if not is_density(rho):
        raise ValueError("Rényi entropy is only defined for density operators.")
    if alpha < 0:
        raise ValueError("Rényi entropy is only defined for positive orders.")
    if alpha == 0:
        return np.log2(np.linalg.matrix_rank(rho))
    if alpha == 1:
        return von_neumann_entropy(rho)

    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 0]

    if alpha == float("inf"):
        return -np.log2(eigs.max())

    return np.log2(pow(eigs, alpha).sum()) / (1 - alpha)
