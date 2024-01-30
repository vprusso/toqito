"""Apply channel a subsystem of an operator."""


import itertools

import numpy as np

from toqito.channel_ops import apply_channel
from toqito.perms import permute_systems
from toqito.states import max_entangled


def partial_channel(
    rho: np.ndarray,
    phi_map: np.ndarray | list[list[np.ndarray]],
    sys: int = 2,
    dim: list[int] | np.ndarray = None,
) -> np.ndarray:
    r"""Apply channel to a subsystem of an operator :cite:`Watrous_2018_TQI`.

    Applies the operator

    .. math::
        \left(\mathbb{I} \otimes \Phi \right) \left(\rho \right).

    In other words, it is the result of applying the channel :math:`\Phi` to the second subsystem
    of :math:`\rho`, which is assumed to act on two subsystems of equal dimension.

    The input :code:`phi_map` should be provided as a Choi matrix.

    This function is adapted from the QETLAB package.

    Examples
    ==========

    The following applies the completely depolarizing channel to the second
    subsystem of a random density matrix.

    >>> from toqito.channel_ops import partial_channel
    >>> from toqito.channels import depolarizing
    >>> rho = np.array([
    ...    [0.3101, -0.0220-0.0219*1j, -0.0671-0.0030*1j, -0.0170-0.0694*1j],
    ...    [-0.0220+0.0219*1j, 0.1008, -0.0775+0.0492*1j, -0.0613+0.0529*1j],
    ...    [-0.0671+0.0030*1j, -0.0775-0.0492*1j, 0.1361, 0.0602 + 0.0062*1j],
    ...    [-0.0170+0.0694*1j, -0.0613-0.0529*1j, 0.0602-0.0062*1j, 0.4530]])
    >>> phi_x = partial_channel(rho, depolarizing(2))
    [[ 0.20545+0.j       0.     +0.j      -0.0642 +0.02495j  0.     +0.j     ]
     [ 0.     +0.j       0.20545+0.j       0.     +0.j      -0.0642 +0.02495j]
     [-0.0642 -0.02495j  0.     +0.j       0.29455+0.j       0.     +0.j     ]
     [ 0.     +0.j      -0.0642 -0.02495j  0.     +0.j       0.29455+0.j     ]]

    The following applies the completely depolarizing channel to the first
    subsystem.

    >>> from toqito.channel_ops import partial_channel
    >>> from toqito.channels import depolarizing
    >>> rho = np.array([[0.3101, -0.0220-0.0219*1j, -0.0671-0.0030*1j, -0.0170-0.0694*1j],
    >>>                 [-0.0220+0.0219*1j, 0.1008, -0.0775+0.0492*1j, -0.0613+0.0529*1j],
    >>>                 [-0.0671+0.0030*1j, -0.0775-0.0492*1j, 0.1361, 0.0602 + 0.0062*1j],
    >>>                 [-0.0170+0.0694*1j, -0.0613-0.0529*1j, 0.0602-0.0062*1j, 0.4530]])
    >>> phi_x = partial_channel(rho, depolarizing(2), 1)
    [[0.2231+0.j      0.0191-0.00785j 0.    +0.j      0.    +0.j     ]
     [0.0191+0.00785j 0.2769+0.j      0.    +0.j      0.    +0.j     ]
     [0.    +0.j      0.    +0.j      0.2231+0.j      0.0191-0.00785j]
     [0.    +0.j      0.    +0.j      0.0191+0.00785j 0.2769+0.j     ]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If Phi map is not provided as a Choi matrix or Kraus
                        operators.
    :param rho: A matrix.
    :param phi_map: The map to partially apply.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If :code:`None`, all dimensions
                are assumed to be equal.
    :return: The partial map :code:`phi_map` applied to matrix :code:`rho`.

    """
    if dim is None:
        dim = np.round(np.sqrt(list(rho.shape))).conj().T * np.ones(2)
    if isinstance(dim, list):
        dim = np.array(dim)

    # Force dim to be a row vector.
    if dim.ndim == 1:
        dim = dim.T.flatten()
        dim = np.array([dim, dim])

    prod_dim_r1 = int(np.prod(dim[0, : sys - 1]))
    prod_dim_c1 = int(np.prod(dim[1, : sys - 1]))
    prod_dim_r2 = int(np.prod(dim[0, sys:]))
    prod_dim_c2 = int(np.prod(dim[1, sys:]))

    if isinstance(phi_map, list):
        # Compute the Kraus operators on the full system.
        s_phi_1, s_phi_2 = len(phi_map), len(phi_map[0])
        phi_list = []
        # Map is completely positive if input is given as:
        # 1. [K1, K2, .. Kr]
        # 2. [[K1], [K2], .. [Kr]]
        # 3. [[K1, K2, .. Kr]] and r > 2
        if isinstance(phi_map[0], np.ndarray):
            phi_list = phi_map
        elif s_phi_2 == 1 or s_phi_1 == 1 and s_phi_2 > 2:
            phi_list = list(itertools.chain(*phi_map))

        if phi_list:
            phi = []
            for m in phi_list:
                phi.append(
                    np.kron(
                        np.kron(np.identity(prod_dim_r1), m),
                        np.identity(prod_dim_r2),
                    )
                )
            phi_x = apply_channel(rho, phi)
        else:
            phi_1 = []
            for m in phi_map:
                phi_1.append(
                    np.kron(
                        np.kron(np.identity(prod_dim_r1), m[0]),
                        np.identity(prod_dim_r2),
                    )
                )
            phi_2 = []
            for m in phi_map:
                phi_2.append(
                    np.kron(
                        np.kron(np.identity(prod_dim_c1), m[1]),
                        np.identity(prod_dim_c2),
                    )
                )

            phi_x = [list(litem) for litem in zip(phi_1, phi_2)]
            phi_x = apply_channel(rho, phi_x)
        return phi_x

    # The `phi_map` variable is provided as a Choi matrix.
    if isinstance(phi_map, np.ndarray):
        dim_phi = phi_map.shape

        dim = np.array(
            [
                [
                    prod_dim_r1,
                    prod_dim_r1,
                    int(dim[0, sys - 1]),
                    int(dim_phi[0] / dim[0, sys - 1]),
                    prod_dim_r2,
                    prod_dim_r2,
                ],
                [
                    prod_dim_c1,
                    prod_dim_c1,
                    int(dim[1, sys - 1]),
                    int(dim_phi[1] / dim[1, sys - 1]),
                    prod_dim_c2,
                    prod_dim_c2,
                ],
            ]
        )
        psi_r1 = max_entangled(prod_dim_r1, False, False)
        psi_c1 = max_entangled(prod_dim_c1, False, False)
        psi_r2 = max_entangled(prod_dim_r2, False, False)
        psi_c2 = max_entangled(prod_dim_c2, False, False)

        phi_map = permute_systems(
            np.kron(np.kron(psi_r1 * psi_c1.conj().T, phi_map), psi_r2 * psi_c2.conj().T),
            [1, 3, 5, 2, 4, 6],
            dim,
        )

        phi_x = apply_channel(rho, phi_map)

        return phi_x

    raise ValueError(
        "The `phi_map` variable is assumed to be provided as "
        "either a Choi matrix or a list of Kraus operators."
    )
