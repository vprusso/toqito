import numpy as np


def  tensor_unravel(constraint_tensor: np.ndarray) -> np.ndarray:
    r"""Convert a tensor-form constraint back to its raw 1D-representation.

    A **tensor-form constraint** is a multi-dimensional numpy array of shape
    :math:`(2, 2, \ldots, 2)` (repeated :math:`n` times). This format is commonly
    used in the context of binary constraint system (BCS) games to represent
    constraints in a consistent and structured way.

    Conceptually, a tensor-form constraint is constructed as follows:

      - The entire array is initially filled with :code:`rhs = (-1)**(b[i])`,
        where :math:`b[i]` is the parity bit corresponding to the row of the
        constraint matrix.
        
      - A single unique element at some index (corresponding to a unique solution)
        is overwritten with :code:`rhs`.

    This function identifies the unique element (i.e. the one that appears exactly once),
    extracts its index, and returns a 1D array of length :math:`n+1`. The first
    :math:`n` entries are the coordinates of the unique index, and the last entry
    is the unique constant :code:`rhs`.
    
    Note:
        This operation is equivalent to **vectorizing** a multi-dimensional
        constraint back into a 1D row of the constraint system matrix.
        It is conceptually related to the standard **vec-operator** for matrices
        (see:cite:`Horn_1985_Matrix`) and to **tensor matricizations**
        in the tensor decomposition literature (see:cite:`Tamara_2009_Tensor`). In this case, 
        since the tensor constraint corresponds to a single constraint row, the operation 
        can also be seen as flattening the row tensor into its 1D representation.
    
    Examples
    ==========
    .. jupyter-execute::
    
       import numpy as np
       from binary_constraint_system_game import tensor_to_raw
        
       tensor_constraint = np.array([[-1, -1], [-1, 1]])
       tensor_to_raw(tensor_constraint)
        
    The tensor-form constraint representation is commonly used in implementations of
    binary constraint system (BCS) games. For background on BCS games, see:cite:`Richard_2014_Characterization`.
    
    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param constraint_tensor: n`-dimensional :code:`numpy` array representing a constraint (shape :code:`(2,)*n`).
    :return: A 1D :code:`numpy` array of length :math:`n+1` where the first :math:`n` elements are the coordinates (indices),
    and the last element is the unique constant (rhs).
    
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
    idx_tuple = tuple(unique_idx[0])
    return np.array(list(idx_tuple) + [unique_value])
