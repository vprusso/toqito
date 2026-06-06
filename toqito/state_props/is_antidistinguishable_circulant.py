"""Check antidistinguishability of circulant pure states via eigenvalue criterion."""

import numpy as np

from toqito.matrix_props.is_circulant import is_circulant


def is_antidistinguishable_circulant(
    gram_matrix: np.ndarray,
    atol: float = 1e-8,
    skip_circulant_check: bool = False,
) -> tuple[bool, float]:
    r"""Check antidistinguishability of a circulant pure-state set via a closed-form eigenvalue criterion.

    For a set of :math:`n` pure states whose Gram matrix :math:`G` (with entries
    :math:`G_{jk} = \langle \psi_j | \psi_k \rangle`) is circulant, determines antidistinguishability
    using Theorem 5.1 of :cite:`johnston2025tight` -- no SDP required.

    The criterion states that the set is antidistinguishable if and only if the eigenvalues
    :math:`\lambda_0 \geq \lambda_1 \geq \cdots \geq \lambda_{n-1}` of :math:`G` satisfy:

    .. math::

        \sqrt{\lambda_0} \leq \sum_{j=1}^{n-1} \sqrt{\lambda_j}

    Args:
        gram_matrix: The :math:`n \times n` circulant Gram matrix of the pure state set.
        atol: Absolute tolerance for eigenvalue comparisons. Defaults to ``1e-8``.
        skip_circulant_check: If ``True``, skips the circulant verification (useful when the
            matrix is known to be circulant). Defaults to ``False``.

    Returns:
        A tuple ``(is_ad, gap)`` where ``is_ad`` is ``True`` if the states are antidistinguishable,
        and ``gap`` is :math:`\sum_{j=1}^{n-1} \sqrt{\lambda_j} - \sqrt{\lambda_0}`
        (non-negative iff antidistinguishable).

    Raises:
        ValueError: If ``gram_matrix`` is not square, not Hermitian, not positive semidefinite,
            or not circulant (when ``skip_circulant_check=False``).

    Examples:
        The trine states have Gram matrix with :math:`-1/2` off-diagonal entries and are
        antidistinguishable (boundary case where gap equals zero):

        >>> import numpy as np
        >>> from toqito.state_props import is_antidistinguishable_circulant
        >>> from toqito.state_props.is_antidistinguishable_circulant import is_antidistinguishable_circulant
        >>> gram = np.array([[1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]])
        >>> is_ad, gap = is_antidistinguishable_circulant(gram)
        >>> bool(is_ad)
        True
        >>> round(gap, 6)
        0.0

    References:
        .. bibliography::
            :filter: docname in docnames

    """
    gram_matrix = np.array(gram_matrix, dtype=complex)

    if gram_matrix.ndim != 2 or gram_matrix.shape[0] != gram_matrix.shape[1]:
        raise ValueError("gram_matrix must be a square 2D array.")

    if not np.allclose(gram_matrix, gram_matrix.conj().T, atol=atol):
        raise ValueError("gram_matrix must be Hermitian.")

    if not skip_circulant_check and not is_circulant(gram_matrix):
        raise ValueError(
            "gram_matrix is not circulant. This criterion only applies to circulant state sets. "
            "Use is_antidistinguishable for the general case."
        )

    # eigvalsh guarantees real eigenvalues for Hermitian matrices.
    eigenvalues = np.linalg.eigvalsh(gram_matrix)

    if np.any(eigenvalues < -atol):
        raise ValueError("gram_matrix is not positive semidefinite.")

    # Clamp tiny negatives from floating point before sqrt.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Sort descending: lambda_0 >= lambda_1 >= ... >= lambda_{n-1}
    eigenvalues = np.sort(eigenvalues)[::-1]

    sqrt_eigs = np.sqrt(eigenvalues)
    lhs = sqrt_eigs[0]
    rhs = float(np.sum(sqrt_eigs[1:]))

    gap = rhs - float(lhs)
    is_ad = bool(gap >= -atol)

    return is_ad, gap