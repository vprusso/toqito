import numpy as np
from toqito.perms.unique_perms import unique_perms
from toqito.matrix.operations.tensor import tensor_list


def calculate_q(Q0, Q1, n, k):
    v = np.ones(n)
    perms = v
    for i in range(k-1):
        v[k-i+1] = 2
        perms = np.array([[perms], [unique_perms(list(v))]])

    mats_0 = []
    mats_1 = []
    for i in range(len(perms)):
        if perms[i] == 1:
            mats_0.append(Q0)
            mats_1.append(Q1)
        if perms[i] == 2:
            mats_0.append(Q1)
            mats_1.append(Q0)

    t_prods_0 = []
    t_prods_1 = []
    for i in range(len(mats_0)):
        t_prods_0 = tensor_list(mats_0)
        t_prods_1 = tensor_list(mats_1)

    Q0_nk = np.concatenate(t_prods_0)
    Q1_nk = np.concatenate(t_prods_1)

    Q0_nk = Q0_nk.reshape((4**n, 4**n))
    Q1_nk = Q1_nk.reshape((4**n, 4**n))

    return Q0_nk, Q1_nk
