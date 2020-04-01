"""Generates a 2-qubit Gisin state."""
import numpy as np


def gisin(lambda_var: float, theta: float) -> np.ndarray:
    r"""
    Produce a Gisin state.

    Returns the Gisin state described in [1].

    Specifically, the Gisin state can be defined as:

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
                                    \end{pmatrix}
        \end{equation}

    References:
    [1] N. Gisin.
        Hidden quantum nonlocality revealed by local filters.
        (http://dx.doi.org/10.1016/S0375-9601(96)80001-6). 1996.

    :param lambda_var: A real parameter in [0, 1].
    :param theta: A real parameter.
    """
    if lambda_var < 0 or lambda_var > 1:
        raise ValueError("InvalidLambda: Variable lambda must be between " "0 and 1.")

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
