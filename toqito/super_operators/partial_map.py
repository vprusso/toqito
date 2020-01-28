import numpy as np


def partial_map(X, Phi, sys=None, dim=None):
    dX = list(X.shape)
    sdX = np.round(np.sqrt(dX))

    if sys is None:
        sys = 2
    if dim is None:
        dim = np.array([[sdX[0], sdX[0]], [sdX[1], sdX[1]]])

    num_sys = len(dim)
