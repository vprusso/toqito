"""Gisin states are the mixtures of separable mixed states and some purely entangled states."""

import numpy as np


def gisin(lambda_var: float, theta: float) -> np.ndarray:
    r"""Produce a Gisin state :cite:`Gisin_1996_Hidden`.

    Returns the Gisin state described in :cite:`Gisin_1996_Hidden`. Specifically, the Gisin state can be defined as:

    .. math::
        \begin{equation}
            \rho_{\lambda, \theta} = \lambda
                                    \begin{pmatrix}
                                        0 & 0 & 0 & 0 \\
                                        0 & \sin^2(\theta) &
                                        -\sin(\theta)\cos(\theta) & 0 \\
                                        0 & -\sin(\theta)\cos(\theta) &
                                        \cos^2(\theta) & 0 \\
                                        0 & 0 & 0 & 0
                                    \end{pmatrix} +
                                    \frac{1 - \lambda}{2}
                                    \begin{pmatrix}
                                        1 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 1
                                    \end{pmatrix}.
        \end{equation}

    Examples
    ==========

    The following code generates the Gisin state :math:`\rho_{0.5, 1}`.

    >>> from toqito.states import gisin
    >>> gisin(0.5, 1)
    array([[ 0.25      ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.35403671, -0.22732436,  0.        ],
           [ 0.        , -0.22732436,  0.14596329,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.25      ]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If :code:`lambda_var` is not a real number.
    :param lambda_var: A real parameter in [0, 1].
    :param theta: A real parameter.
    :return: Gisin state.

    """
    if lambda_var < 0 or lambda_var > 1:
        raise ValueError("InvalidLambda: Variable lambda must be between 0 and 1.")

    rho_theta = np.array(
        [
            [0, 0, 0, 0],
            [0, np.sin(theta) ** 2, -np.sin(2 * theta) / 2, 0],
            [0, -np.sin(2 * theta) / 2, np.cos(theta) ** 2, 0],
            [0, 0, 0, 0],
        ]
    )

    rho_uu_dd = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

    return lambda_var * rho_theta + (1 - lambda_var) * rho_uu_dd / 2
