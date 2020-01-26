import numpy as np
from numpy import linalg as LA
import cvxpy as cvx
from toqito.helper.constants import e0, e1, e00, e11
from toqito.matrix.operations.tensor import tensor_list
from collections import defaultdict
from toqito.states.w_state import w_state
from toqito.states.bell import bell
from toqito.helper.unique_perms import unique_perms
from toqito.matrix.operations.vec import vec
from toqito.helper.iden import iden
from scipy.sparse import csr_matrix, lil_matrix
import operator
import functools
from scipy import sparse
from toqito.states.ghz_state import ghz_state
from toqito.hedging.pi_perm import pi_perm
from typing import List
from scipy.sparse import issparse
from toqito.helper.swap import swap
from toqito.helper.swap_operator import swap_operator
from toqito.hedging.calculate_hedging_value import calculate_hedging_value

def get_mat_entry_bin(A, b):
    block_bin = b[0:2]
    elem_bin = b[2:]
    
    if block_bin == "00":
        block = A[0:2, 0:2]
    elif block_bin == "01":
        block = A[0:2, 2:]
    elif block_bin == "10":
        block = A[2:, 0:2]
    elif block_bin == "11":
        block = A[2:, 2:]

    pos = list(map(int, list(elem_bin)))
    val = block[pos[0], pos[1]]
    return val


def bin_perm(entry_int, dim):
    b = bin(entry_int)[2:].zfill(dim)
    print(f"BEFORE BIN: {b}")
    return b[0] + b[1] + b[3] + b[2]




def main():
    n = 2
    k = 1
    alpha = 1/np.sqrt(2)
    theta = np.pi/8

   # calculate_hedging_value(n, k, alpha, theta)

    u = alpha * (e00 + e11)
    v = np.cos(theta)*e00 + np.sin(theta)*e11
    P1 = v * v.conj().T
    P0 = np.identity(4) - P1

    rho = u * u.conj().T
    Psi = swap(rho)

    #print(P0)


    A = np.kron(e0*e0.conj().T, e1*e1.conj().T)
    Psi_A = swap(A)

    rho = 1/2*np.kron(e0*e0.conj().T, e0*e0.conj().T) + \
        1/2*np.kron(e0*e1.conj().T, e0*e1.conj().T) + \
        1/2*np.kron(e1*e0.conj().T, e1*e0.conj().T) + \
        1/2*np.kron(e1*e1.conj().T, e1*e1.conj().T)
    #print(x)

    y = 1/2*(e00*e00.conj().T + e00*e11.conj().T + e11*e00.conj().T + e11*e11.conj().T)
    print(y)

    mat = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])
    count = 0
    dim = 4
    Z = P0
    A = rho
    for i in range(dim):
        for j in range(dim):
            Z_val = Z[i][j]
            perm_bin = bin_perm(count, dim)
            perm_val = rho[int(perm_bin[:2], 2), int(perm_bin[2:], 2)]
            print(Z_val, perm_val)
            mat[i,j] = Z_val * perm_val
            print(f"BEFORE VAL: {Z_val}")
            print(f"AFTER BIN: {perm_bin}")
            print(f"AFTER VAL: {perm_val}")
            print(Z_val * perm_val)
            count += 1
#    get_mat_entry_bin(mat, "1100")
    
    print(mat)
    print(P0)
    z = np.sin(np.pi/8)**2
    print(z*1/2)

#    y = 1/2*np.kron(e0*e0.conj().T)



    #Q0 = np.kron(np.identity(1), Psi) * P0
    #Q1 = np.kron(np.identity(1), Psi) * P1

    #print(Q0)
    #print(Q1)

#    calculate_hedging_value(n, k, alpha, theta)


if __name__ == "__main__":
    main()

