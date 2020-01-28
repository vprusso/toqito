import numpy as np
from numpy import linalg as LA
import cvxpy as cvx
import itertools
from toqito.helper.constants import e0, e1, e00, e11
from toqito.matrix.operations.tensor import tensor_list
from collections import defaultdict
from toqito.states.w_state import w_state
from toqito.states.bell import bell
from toqito.perms.unique_perms import unique_perms
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
from toqito.perms.swap import swap
from toqito.perms.swap_operator import swap_operator
from toqito.hedging.calculate_hedging_value import calculate_hedging_value
from toqito.states.pure_to_mixed import pure_to_mixed
from toqito.states.purity import purity
from numpy.linalg import matrix_power
from toqito.super_operators.choi_map import choi_map
from toqito.super_operators.reduction_map import reduction_map
from toqito.super_operators.partial_trace import partial_trace
from toqito.super_operators.apply_map import apply_map
from toqito.super_operators.partial_transpose import partial_transpose

X = np.array([[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
Y = np.array([[12, 14], [20, 22]])

print(partial_trace(X,[1,2]))

#print(all(x == 1 for x in itertools.chain(*bool_mat)))

