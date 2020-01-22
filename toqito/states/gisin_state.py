import numpy as np


def gisin_state(lambda_var: float, theta: float) -> np.ndarray:
    """
    Produces a Gisin state.
    :param lambda_var: A real parameter in [0, 1].
    :param theta: A real parameter.

    Returns the Gisin state described in [1].

    The Gisin states are a mixture of the entangled state rho_theta and the
    separable states rho_uu and rho_dd.

    References:
    [1] N. Gisin. Hidden quantum nonlocality revealed by local filters.
        (http://dx.doi.org/10.1016/S0375-9601(96)80001-6). 1996.

    """
    if lambda_var < 0 or lambda_var > 1:
        raise ValueError("ValueError: variable lambda must be between 0 and 1.")

    rho_theta = np.array([[0, 0, 0, 0],
                          [0, np.sin(theta)**2, -np.sin(2*theta)/2, 0],
                          [0, -np.sin(2*theta)/2, np.cos(theta)**2, 0],
                          [0, 0, 0, 0]])

    rho_uu_dd = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]])

    return lambda_var * rho_theta + (1-lambda_var) * rho_uu_dd/2
