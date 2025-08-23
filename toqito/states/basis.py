"""A basis state represents the standard basis vectors of some n-dimensional Hilbert Space.

Here, n can be given as a parameter as shown below.
"""

import numpy as np


def basis(dim: int, pos: int) -> np.ndarray:
    r"""Obtain the ket of dimension :code:`dim` :footcite:`WikiBraKet`.

    Examples
    ==========

    The standard basis ket vectors given as :math:`|0 \rangle` and :math:`|1 \rangle` where

    .. math::
        |0 \rangle = \left[1, 0 \right]^{\text{T}} \quad \text{and} \quad
        |1 \rangle = \left[0, 1 \right]^{\text{T}},

    can be obtained in :code:`|toqito⟩` as follows.

    Example:  Ket basis vector: :math:`|0\rangle`.

    .. jupyter-execute::

        from toqito.states import basis
        basis(2, 0)

    Example: Ket basis vector: :math:`|1\rangle`.

    .. jupyter-execute::

        from toqito.states import basis
        basis(2, 1)

    References
    ==========
    .. footbibliography::



    :raises ValueError: If the input position is not in the range [0, dim - 1].
    :param dim: The dimension of the column vector.
    :param pos: 0-indexed position of the basis vector where the 1 will be placed.
    :return: The column vector of dimension :code:`dim` with all entries set to `0` except the entry
             at `pos` which is set to `1`.

    """
    if pos >= dim or pos < 0:
        raise ValueError("Invalid: The `pos` variable needs to be between [0, dim - 1] for ket function.")

    ret = np.zeros(dim, dtype=np.int64)
    ret[pos] = 1
    return ret.reshape(-1, 1)
