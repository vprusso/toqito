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
from toqito.states.pure_to_mixed import pure_to_mixed
from toqito.states.purity import purity
from numpy.linalg import matrix_power
from toqito.super_operators.choi_map import choi_map
from toqito.super_operators.reduction_map import reduction_map
from toqito.super_operators.partial_trace import partial_trace
from toqito.super_operators.apply_map import apply_map

X = np.array([[1, 2],
              [3, 4]])

K1 = np.array([[1, 5], [1, 0], [0, 2]])
K2 = np.array([[0, 1], [2, 3], [4, 5]])
K3 = np.array([[-1, 0], [0, 0], [0, -1]])
K4 = np.array([[0, 0], [1, 1], [0, 0]])

#print(apply_map(X, swap_operator(3)))
print(apply_map(X, [[K1, K2], [K3, K4]]))


#X = np.array([[1, 2, 3, 4],
#              [5, 6, 7, 8],
#              [9, 10, 11, 12],
#              [13, 14, 15, 16]])

#print(partial_trace(X))
