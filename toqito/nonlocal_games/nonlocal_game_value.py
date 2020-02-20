from numpy.matlib import repmat
import numpy as np

from toqito.perms.permute_systems import permute_systems


def nonlocal_game_value(p, V, strategy: str = "classical", k=1, rept=1):

    ma, mb = p.shape[0], p.shape[1]
    oa, ob = V.shape[0], V.shape[1]

    if strategy == "nosignal":
        pass

    if strategy == "quantum":
        pass

    if strategy == "classical":
        ngval = float("-inf")
        a_ind = np.zeros((1, ma))[0]
        b_ind = np.zeros((1, mb))[0]
        a_ind[ma-1] = oa-2
        b_ind[mb-1] = ob-2

        x = repmat(p, oa, ob)
        print(x.shape)
#        y = np.transpose(x, (2, 3, 0, 1))

#        x = np.tile(p[:, 1], [1, 1, oa, ob])
#        x = np.tile(p, (1, 1, oa, ob))
#        print(x.shape)
#        print(x)
#        print(p)

