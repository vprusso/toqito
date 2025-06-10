"""Checks if a quantum state is absolutely PPT."""

import numpy as np

from toqito.matrix_props import is_square, is_positive_semidefinite
from toqito.state_props import in_separable_ball, abs_ppt_constraints


def is_abs_ppt(mat: np.ndarray, dim: int | list[int] = None) -> int:
    r"""Determine whether or not a matrix is absolutely PPT :cite:`ref`.

        This function is adapted from QETLAB.

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
        :param dim: A 1-by-2 vector containing the dimensions of the
                    subsystems on which :code:`mat` acts.
        :return: :code:`1` if :code:`mat` is absolutely PPT,
        :return: :code:`0` if :code:`mat is not absolutely PPT,
        :return: :code:`-1` if the function could not decide.

    """
    if not is_square(mat):
        raise ValueError("Matrix must be square")

    nm = mat.shape[0]

    if dim is None:
        # Find the largest divisor d of nm such that d ** 2 <= nm
        # nm won't be too large, so let's just use a for-loop
        # Floating-point arithmetic is risky
        d = 1
        for j in range(1, nm + 1):
            if j ** 2 > nm:
                break
            if nm % j == 0:
                d = j
        dim = [j, nm // j]

    n, m = dim
    p = min(n, m)
    if not n * m == nm:
        raise ValueError("Product of dim[0] and dim[1] must equal dimensions of mat")

    # Quick checks
    # 1. Is Hermitian
    if not is_hermitian(mat):
        return False
    # Compute eigenvalues (in descending order)
    eigs = np.linalg.eigvalsh(mat)[::-1]
    # 2. Is PSD (TODO: use tolerances)
    if eigs[-1] < 0:
        return 0
    # 3. Check if mat is in separable ball
    if in_separable_ball(mat):
        return 1
    # 4. Check Theorem 7.2 of arXiv:1406.1277
    if sum(eigs[:p-1]) <= eigs[-1] + sum(eigs[-p:]):
        return 1

    # Main check
    # 33592 is the upper bound on the number of constraint matrices that is required
    # for p = 6 without any additional checks in abs_ppt_constraints
    # Reference:
    # http://njohnston.ca/2014/02/counting-the-possible-orderings-of-pairwise-multiplication/
    # 2608 is the minimum value: The list of optimal constraint counts can be found
    # in https://oeis.org/A237749
    constraints = abs_ppt_constraints(eigs, p, 33592)
    for constraint in constraints:
        if not is_positive_semidefinite(constraint):
            return 0
    # We checked all constraints for p <= 6, but not for p >= 7
    return 1 if p <= 6 else -1
