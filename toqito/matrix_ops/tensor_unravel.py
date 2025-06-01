import numpy as np

def  tensor_unravel(constraint_tensor: np.ndarray) -> np.ndarray:
    r"""Convert a tensor-form constraint back to its raw 1D representation.
    The tensor-form constraint is expected to have shape (2, 2, ..., 2) (n times)
    and to be constructed as follows:
      - The array is initially filled with -rhs, where rhs = (-1)**(b[i]).
      - A unique element at some index is overwritten with rhs.
    This function finds the unique element (the one that appears exactly once)
    and returns a 1D array of length n+1, where the first n entries are the coordinates
    (taken directly from the unique index) and the last element is the unique constant rhs.
    Examples
    ==========
        >>> import numpy as np
        >>> from binary_constraint_system_game import tensor_to_raw
        >>> tensor_constraint = np.array([[-1, -1], [-1, 1]])
        >>> raw = tensor_to_raw(tensor_constraint)
        >>> raw
        array([1, 1, 1])
    :param constraint_tensor: An n-dimensional NumPy array representing a constraint (shape (2,)*n).
    :return: A 1D NumPy array of length n+1 where the first n elements are the coordinates (indices)
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
    raw_constraint = np.array(list(idx_tuple) + [unique_value])
    return raw_constraint
