"""Check if state has NPT (negative partial transpose) criterion."""


import numpy as np

from toqito.state_props import is_ppt


def is_npt(mat: np.ndarray, sys: int = 2, dim: int | list[int] = None, tol: float = None) -> bool:
    r"""Determine whether or not a matrix has negative partial transpose :cite:`WikiPeresHorodecki`.

    Yields either :code:`True` or :code:`False`, indicating that :code:`mat` does or does not have
    negative partial transpose (within numerical error). The variable :code:`mat` is assumed to act
    on bipartite space. :cite:`DiVincenzo_2000_Evidence`

    A state has negative partial transpose if it does not have positive partial transpose.

    Examples
    ========
    To check if a matrix has negative partial transpose

    >>> import numpy as np
    >>> from toqito.state_props import is_npt
    >>> from toqito.states import bell
    >>> is_npt(bell(2) @ bell(2).conj().T, 2)
    True


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param mat: A square matrix.
    :param sys: Scalar or vector indicating which subsystems the transpose
                should be applied on. Default value is `2`.
    :param dim: The dimension is a vector containing the dimensions of the
                subsystems on which :code:`mat` acts.
    :param tol: Tolerance with which to check whether `mat` is PPT.
    :return: Returns :code:`True` if :code:`mat` is NPT and :code:`False` if
             not.

    """
    return not is_ppt(mat, sys, dim, tol)
