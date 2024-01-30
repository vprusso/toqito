"""Compute the diamond norm between two quantum channels."""
import cvxpy
import numpy as np


def diamond_norm(choi_1: np.ndarray, choi_2: np.ndarray) -> float:
    r"""Return the diamond norm distance between two quantum channels.

    The calculation uses the simplified semidefinite program of Watrous in
    :cite:`Watrous_2009_Semidefinite`.

    This function has been adapted from :cite:`Rigetti_2022_Forest`

    .. note::
        This calculation becomes very slow for 4 or more qubits.


    Examples
    ========
    Consider the depolarizing and identity channels in a 2-dimensional space. The depolarizing channel parameter is
    set to 0.2:

    >>> import numpy as np
    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_metrics import diamond_norm
    >>> choi_depolarizing = depolarizing(dim=2, param_p=0.2)
    >>> choi_identity = np.identity(2**2)
    >>> dn = diamond_norm(choi_depolarizing, choi_identity)
    >>> print("Diamond norm between depolarizing and identity channels: ", dn)
    Diamond norm between depolarizing and identity channels:  -2.1680424534747078e-07

    Similarly, we can compute the diamond norm between the dephasing channel (with parameter 0.3) and the identity
    channel:

    >>> import numpy as np
    >>> from toqito.channels import dephasing
    >>> from toqito.channel_metrics import diamond_norm
    >>> choi_dephasing = dephasing(dim=2)
    >>> choi_identity = np.identity(2**2)
    >>> dn = diamond_norm(choi_dephasing, choi_identity)
    >>> print("Diamond norm between dephasing and identity channels: ", dn)
    Diamond norm between depolarizing and identity channels:  -3.5989938811978797e-09

    References
    ==========
        .. bibliography::
            :filter: docname in docnames


    :raises ValueError: If matrices are not of equal dimension.
    :raises ValueError: If matrices are not square.
    :param choi_1: A 4**N by 4**N matrix (where N is the number of qubits).
    :param choi_2: A 4**N by 4**N matrix (where N is the number of qubits).

    """
    if choi_1.shape != choi_2.shape:
        raise ValueError("The Choi matrices provided should be of equal dimension.")

    choi_dim_x, choi_dim_y = choi_1.shape
    if choi_dim_x != choi_dim_y:
        raise ValueError("The Choi matrix provided must be square.")

    dim_squared = choi_1.shape[0]
    dim = int(np.sqrt(dim_squared))

    delta_choi = choi_1 - choi_2

    # Enforce Hermiticity.
    delta_choi = (delta_choi.conj().T + delta_choi) / 2

    # Enforce that variable is density operator.
    rho = cvxpy.Variable([dim, dim], complex=True)
    constraints = [rho == rho.H]
    constraints += [rho >> 0]
    constraints += [cvxpy.trace(rho) == 1]

    # Variable must be Hermitian and positive-semidefinite.
    w_var = cvxpy.Variable([dim_squared, dim_squared], complex=True)
    constraints += [w_var == w_var.H]
    constraints += [w_var >> 0]

    constraints += [(w_var - cvxpy.kron(np.eye(dim), rho)) << 0]

    j_var = cvxpy.Parameter([dim_squared, dim_squared], complex=True)
    objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(j_var.H @ w_var)))

    problem = cvxpy.Problem(objective, constraints)
    j_var.value = delta_choi
    problem.solve()

    return problem.value * 2
