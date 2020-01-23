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
from scipy.sparse import csr_matrix, lil_matrix
import operator
import functools
from scipy import sparse
from toqito.states.ghz_state import ghz_state

def main():
    x = ghz_state(2, 3)
    print(x[7][0])
#    print(ghz_state(3, 10))

if __name__ == "__main__":
    main()

