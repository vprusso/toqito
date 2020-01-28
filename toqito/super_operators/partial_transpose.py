import numpy as np
from typing import List
from toqito.helper.permute_systems import permute_systems


def partial_transpose(X: np.ndarray,
                      sys: int = 2,
                      dim: List[int] = None) -> np.ndarray:
    """
    Computes the partial transpose of a matrix.

    :param X: A matrix.
    :returns: The partial transpose of matrix X.

    By default, the returned matrix is the partial transpose of the matrix X,
    where it is assumed that the number of rows and columns of X are both
    perfect squares and both subsystems have equal dimension. The transpose is
    applied to the second subsystem.

    In the case where SYS amd DIM are specified, this function gives the
    partial transpose of the matrix X where the dimensions of the (possibly
    more than 2) subsystems are given by the vector DIM and the subsystems to
    take the partial transpose are given by the scalaer or vector SYS. If X is
    non-square, different row and column dimensions can be specified by putting
    the row dimensions in the first row of DIM and the column dimensions in the
    second row of DIM.
    """
    eps = np.finfo(float).eps
    dX = list(X.shape)
    sdX = np.round(np.sqrt(dX))

    if dim is None:
        dim = np.array([[sdX[0], sdX[0]],
                        [sdX[1], sdX[1]]])

    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim, dX[0]/dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2*dX[0]*eps:
            msg = """
                InvalidDim: If DIM is a scalar, X must be square and DIM must
                evenly divide length(X); please provide the DIM array containing
                the dimensions of the subsystems.
            """
            raise ValueError(msg)
            dim[1] = np.round(dim[1])
            num_sys = 2

    # Allow the user to enter a vector for dim if X is square.
    if min(X.shape) == 1:
        # Force dim to be a row vector.
        dim = dim[:]
        dim = np.array([[dim], [dim]])

    # Prepare the partial transposition.
    prod_dimR = np.prod(dim[0][:])
    prod_dimC = np.prod(dim[1][:])
    sub_prodR = np.prod(dim[0][sys-1])
    sub_prodC = np.prod(dim[1][sys-1])
    sub_sys_vecR = prod_dimR * np.ones(int(sub_prodR)) / sub_prodR
    sub_sys_vecC = prod_dimC * np.ones(int(sub_prodC)) / sub_prodC

    s1 = list(range(1, num_sys+1))
    s2 = [sys]
    set_diff = list(set(s1) - set(s2))
    
    perm = [sys]
    perm.extend(set_diff)

    # Permute the subsystems so that we just have to do the partial transpose
    # on the first (potentially larger) subsystem.
    Xpt = permute_systems(X, perm, dim)

    A = np.reshape(
            Xpt,
            (int(sub_sys_vecR[0]),
                int(sub_prodR),
                int(sub_sys_vecC[0]),
                int(sub_prodC)),
            order="F")
    
    B = np.transpose(A, [0, 3, 2, 1])

    Xpt = np.reshape(B, (int(prod_dimR), int(prod_dimC)), order="F")

    # Return the subsystems back to their original positions.
    dim[:][sys-1] = dim[[1, 0], sys-1]
    perm_np = np.array(perm)
    perm_np = list(perm_np - 1)
    dim = dim[:][perm_np]

    return permute_systems(Xpt, perm, dim, False, True)

