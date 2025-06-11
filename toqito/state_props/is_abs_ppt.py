"""Checks if a quantum state is absolutely PPT."""

import numpy as np

from toqito.matrix_props import is_hermitian, is_positive_semidefinite, is_square
from toqito.state_props.abs_ppt_constraints import abs_ppt_constraints
from toqito.state_props.in_separable_ball import in_separable_ball


def is_abs_ppt(mat: np.ndarray, dim: int = None, rtol: float = 1e-05, atol: float = 1e-08) -> bool | None:
    r"""Determine whether or not a matrix is absolutely PPT :cite:`Hildebrand_2007_AbsPPT`.

    This function is adapted from QETLAB :cite:`QETLAB_link`.

    Examples
    ==========
    A random density matrix will likely not be absolutely PPT:

    .. jupyter-execute::

        import numpy as np
        from toqito.rand import random_density_matrix
        from toqito.state_props import is_abs_ppt, is_separable
        rho = random_density_matrix(9) # assumed to act on a 3 x 3 bipartite system
        print("rho is absolutely PPT:", is_abs_ppt(rho, 3))

    The maximally-mixed state is an example of an absolutely PPT state:

    .. jupyter-execute::

        import numpy as np
        from toqito.states import max_mixed
        from toqito.state_props import is_abs_ppt, is_separable
        rho = max_mixed(9) # assumed to act on a 3 x 3 bipartite system
        print("rho is absolutely PPT:", is_abs_ppt(rho, 3))

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If :code:`mat` is not square.
    :raises ValueError: If :code:`dim` does not divide the dimensions of :code:`mat`.
    :param mat: A square matrix.
    :param dim: The dimension of any one subsystem on which :code:`mat` acts. If :code:`None`,
                :code:`dim` is selected such that :code:`min(dim, mat.shape[0] // dim)` is
                maximised, since this gives the strongest conditions on being absolutely PPT
                (see Theorem 2 of :cite:`Hildebrand_2007_AbsPPT`).
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: :code:`True` if :code:`mat` is absolutely PPT, :code:`False` if :code:`mat is not
             absolutely PPT, and :code:`None` if the function could not decide.
    :notes: If :code:`min(dim)` :math:`\leq 6`, this function checks all constraints
            and therefore returns :code:`True` or :code:`False` in all cases. However, if
            :code:`min(dim)` :math:`\geq 7`, only the first :math:`33592` constraints are
            checked, since there are over :math:`23178480` constraints in this case
            :cite:`Johnston_2014_Orderings`. Therefore the function returns either
            :code:`False` if at least one constraint was not satisfied, or :code:`None`
            if all checked constraints were satisfied.

    """
    if not is_square(mat):
        raise ValueError("Matrix must be square")

    nm = mat.shape[0]

    if dim is None:
        # Find the largest divisor d of nm such that d ** 2 <= nm
        # nm won't be too large, so let's just use a for-loop
        # Floating-point arithmetic is risky
        dim = 1
        for j in range(1, nm + 1):
            if j**2 > nm:
                break
            if nm % j == 0:
                dim = j

    if nm % dim != 0:
        raise ValueError("dim must divide the dimensions of the matrix")

    n, m = dim, nm // dim
    p = min(n, m)

    # Quick checks
    # 1. Is Hermitian
    if not is_hermitian(mat, rtol, atol):
        return False
    # Compute eigenvalues (in descending order)
    # eigsvalsh normally returns eigenvalues in ascending order
    # But it is risky to assume this will remain the default behaviour in the future
    eigs = np.sort(np.linalg.eigvalsh(mat))[::-1]
    # 2. Is PSD
    if eigs[-1] < -abs(atol):
        return False
    # 3. Check Theorem 7.2 of :cite:`Jivulescu_2015_Reduction`
    if sum(eigs[: p - 1]) <= eigs[-1] + sum(eigs[-p:]):
        return True
    # 4. Check if mat is in separable ball
    if in_separable_ball(mat):
        return True

    # Main check
    constraints = abs_ppt_constraints(eigs, p)
    for constraint in constraints:
        if not is_positive_semidefinite(constraint, rtol, atol):
            return False
    # We checked all constraints for p <= 6, but not for p >= 7
    return True if p <= 6 else None
