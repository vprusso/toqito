"""Checks if a quantum state is absolutely PPT."""

import numpy as np

from toqito.matrix_props import is_square, is_positive_semidefinite
from toqito.state_props import in_separable_ball, abs_ppt_constraints


def is_abs_ppt(mat: np.ndarray, dim: int = None) -> int:
    r"""Determine whether or not a matrix is absolutely PPT :cite:`Hildebrand_2007_AbsPPT`.

        This function is adapted from QETLAB :cite:`QETLAB_link`.

        :notes: If :code:`min(dim)` is :math:`\leq 6`, this function checks all constraints
        and therefore returns :code:`True` or :code:`False` in all cases. However, if
        :code:`min(dim)` is :math:`\geq 7`, only the first :math:`33592` constraints are
        checked, since there are over :math:`23178480` constraints in this case
        :cite:`Johnston_2014_Orderings`. Therefore the function returns either :code:`False`,
        or :code:`None` if all checked constraints were satisfied.

        Examples
        ==========
        Demonstrate how the function works with expected output.

        .. jupyter-execute::

            import numpy as np
            x = np.array([[1, 2], [3, 4]])
            print(x)

        References
        ==========
        .. bibliography::
            :filter: docname in docnames

        :param mat: A square matrix.
        :param dim: The dimension of any one subsystem on which :code:`mat` acts.
        :return: :code:`True` if :code:`mat` is absolutely PPT,
        :return: :code:`False` if :code:`mat is not absolutely PPT,
        :return: :code:`None` if the function could not decide.

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
            if j ** 2 > nm:
                break
            if nm % j == 0:
                dim = j

    if nm % dim != 0:
        raise ValueError("dim must divide the dimensions of the matrix")

    n, m = dim, nm // dim
    p = min(n, m)

    # Quick checks
    # 1. Is Hermitian
    if not is_hermitian(mat):
        return False
    # Compute eigenvalues (in descending order)
    # eigsvalsh normally returns eigenvalues in ascending order
    # But it is risky to assume this will remain the default behaviour in the future
    eigs = np.sort(np.linalg.eigvalsh(mat))[::-1]
    # 2. Is PSD (TODO: use tolerances)
    if eigs[-1] < 0:
        return False
    # 3. Check if mat is in separable ball
    if in_separable_ball(mat):
        return True
    # 4. Check Theorem 7.2 of arXiv:1406.1277
    if sum(eigs[:p-1]) <= eigs[-1] + sum(eigs[-p:]):
        return True

    # Main check
    constraints = abs_ppt_constraints(eigs, p)
    for constraint in constraints:
        if not is_positive_semidefinite(constraint):
            return False
    # We checked all constraints for p <= 6, but not for p >= 7
    return True if p <= 6 else None
