"""Generate the 1D contraint."""

import numpy as np


def tensor_unravel(constraint_tensor: np.ndarray) -> np.ndarray:
    r"""Decode a clause tensor (indicator tensor) into its raw 1D representation.

    In binary constraint system (BCS) games, parity constraints can be encoded as
    **clause tensors** — n-dimensional NumPy arrays of shape `(2, 2, ..., 2)`,
    filled with a constant background value (e.g., `(-1)**b[i]`) except for a
    single unique entry that marks the satisfying assignment.

    This function unravels such a tensor by:
       1. Locating the unique element (the one appearing exactly once).
       2. Extracting its multi-dimensional index `(i1, i2, ..., in)`.
       3. Returning a 1D NumPy array `[i1, i2, ..., in, value]`, where the first `n`
          entries are the coordinates and the last entry is the unique value (±1).

    Conceptually, this is a form of structured tensor decoding, closely related to:

    - Indicator (Kronecker delta) tensors in multilinear algebra refer to :footcite:`Kolda_2009_Tensor`.
    - The matrix ``vec``-operator for flattening matrices refer to :footcite:`Horn_1985_Matrix`.
    - Parity-projector encodings in linear-system games refer to :footcite:`Cleve_2016_Perfect`.

    The tensor-form constraint representation is commonly used in implementations of
    binary constraint system (BCS) games. For background on BCS games, refer to :footcite:`Cleve_2014_Characterization`.

    Examples
    ==========
    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import tensor_unravel

     tensor_constraint = np.array([[-1, -1], [-1, 1]])
     tensor_unravel(tensor_constraint)

    References
    ==========
    .. footbibliography::

    :param constraint_tensor: An n-dimensional tensor with shape `(2,)*n`, where each element is either -1 or +1.
                              All entries should be equal except for one unique position that marks
                              the satisfying assignment.
    :return: A 1D :code:`numpy` array of length :math:`n+1` where the first :math:`n`
                    elements are the coordinates (indices), and the last element is the unique constant (rhs).

    """
    values, counts = np.unique(constraint_tensor, return_counts=True)
    if len(values) != 2:
        raise ValueError("Constraint tensor does not have exactly two distinct values.")
    if counts[0] == 1:
        unique_value = values[0]
    elif counts[1] == 1:
        unique_value = values[1]
    else:
        raise ValueError("Constraint tensor does not have a unique element that appears exactly once.")
    unique_idx = np.argwhere(constraint_tensor == unique_value)
    if unique_idx.shape[0] != 1:
        raise ValueError("Expected exactly one occurrence of the unique value in the constraint tensor.")
    return np.array(list(unique_idx[0]) + [unique_value])
