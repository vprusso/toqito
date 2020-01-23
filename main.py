import numpy as np
from numpy import linalg as LA
import cvxpy as cvx
from toqito.helper.constants import e0, e1
from toqito.matrix.operations.tensor import tensor_list
from collections import defaultdict
from toqito.states.w_state import w_state
from toqito.states.bell import bell
from toqito.helper.unique_perms import unique_perms
from toqito.matrix.operations.vec import vec
from toqito.helper.iden import iden
import operator
import functools
from scipy import sparse


def max_entangled(dim: int, is_sparse: bool = False, is_normalized: bool = True):
    psi = np.reshape(iden(dim, is_sparse), (dim**2, 1))
    if is_normalized:
        psi = psi/np.sqrt(dim)
    return psi


def main():
    psi = max_entangled(2)
    print(psi)

    x = 1/np.sqrt(2) * (np.kron(e0, e0) + np.kron(e1, e1))
    
    print(x)

if __name__ == "__main__":
    main()

