import numpy as np
from toqito.perms.unique_perms import unique_perms
from toqito.matrix.operations.tensor import tensor_list


def calculate_q(Q0, Q1, n, k):
    v = np.ones(n)
    perms = np.ones(n)
    for i in range(k-1):
        v[k-i-1] = 2
        uniq_perms = list(unique_perms(list(v)))
        perms = np.vstack((perms, uniq_perms))

    # This is just a 1-D vector
    if len(perms.shape) == 1:
        perms = np.array([perms])
        r, c = perms.shape[0], perms.shape[1]
    # This is a N-D matrix.
    else:
        r, c = perms.shape[0], perms.shape[1]

    mats_0 = [[None]*c]*r
    mats_1 = [[None]*c]*r
    for i in range(r):
        for j in range(c):
#            print(perms[i][j])
            if perms[i][j] == 1:
                mats_0[i][j] = Q0
                mats_1[i][j] = Q1
            if perms[i][j] == 2:
                mats_0[i][j] = Q1
                mats_1[i][j] = Q0
#            print(i, j, mats_0[i][j])

    print(mats_0[0][0])
    print(mats_0[0][1])
    t_prods_0 = []
    t_prods_1 = []

    for i in range(len(mats_0)):
        t_prods_0.append(tensor_list(mats_0[i]))
        t_prods_1.append(tensor_list(mats_1[i]))

    Q0_nk = np.concatenate(t_prods_0)
    Q1_nk = np.concatenate(t_prods_1)

    # print(Q0_nk.shape)
    # print(Q0_nk.size)
    #
    # Q0_nk = Q0_nk.reshape((4**n, 4**n))
    # Q1_nk = Q1_nk.reshape((4**n, 4**n))
    #
    # return Q0_nk, Q1_nk
