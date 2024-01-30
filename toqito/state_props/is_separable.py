"""Check if state is separable."""


import numpy as np
from picos import partial_trace

from toqito.channels import realignment
from toqito.matrix_props import is_positive_semidefinite, trace_norm
from toqito.state_props import in_separable_ball, is_ppt
from toqito.state_props.has_symmetric_extension import has_symmetric_extension


def is_separable(
    state: np.ndarray, dim: None | int | list[int] = None, level: int = 2, tol: float = 1e-8
) -> bool:
    r"""Determine if a given state (given as a density matrix) is a separable state :cite:`WikiSepSt`.

    Examples
    ==========
    Consider the following separable (by construction) state:

    .. math::
        \rho = \rho_1 \otimes \rho_2.
        \rho_1 = \frac{1}{2} \left(
            |0 \rangle \langle 0| + |0 \rangle \langle 1| + |1 \rangle \langle 0| + |1 \rangle \langle 1| \right)
        \rho_2 = \frac{1}{2} \left( |0 \rangle \langle 0| + |1 \rangle \langle 1| \right)

    The resulting density matrix will be:

    .. math::
        \rho =  \frac{1}{4} \begin{pmatrix}
                1 & 0 & 1 & 0 \\
                0 & 1 & 0 & 1 \\
                1 & 0 & 1 & 0 \\
                0 & 1 & 0 & 1
                \end{pmatrix} \in \text{D}(\mathcal{X}).

    We provide the input as a denisty matrix :math:`\rho`.

    On the other hand, a random density matrix will be an entangled state (a separable state).
    >>> from toqito.rand import random_density_matrix
    >>> from toqito.state_props import is_separable
    >>> rho_separable = np.array([[1, 0, 1, 0],
    ...                           [0, 1, 0, 1],
    ...                           [1, 0, 1, 0],
    ...                           [0, 1, 0, 1]])
    >>> rho_random = random_density_matrix(4)
    >>> is_separable(rho_separable)
    True
    >>> is_separable(rho_random)
    False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If dimension is not specified.
    :param state: The matrix to check.
    :param dim: The dimension of the input.
    :param level: The level up to which to search for the symmetric extensions.
    :param tol: Numerical tolerance used.
    :return: :code:`True` if :code:`rho` is separabale and :code:`False` otherwise.

    """
    if not is_positive_semidefinite(state):
        raise ValueError("Checking separability of non-positive semidefinite matrix is invalid.")

    state_len = state.shape[1]
    state_rank = np.linalg.matrix_rank(state)
    state = state / np.trace(state)
    eps = np.finfo(float).eps

    if dim is None:
        dim = int(np.round(np.sqrt(state_len)))

    if isinstance(dim, int):
        dim = np.array([dim, state_len / dim])  # pylint: disable=redefined-variable-type
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * state_len * eps:
            raise ValueError("The parameter `dim` must evenly divide the length of the state.")
        dim[1] = np.round(dim[1])

    # nD, xD, pD
    min_dim = int(min(dim))
    max_dim = int(max(dim))
    prod_dim = int(np.prod(dim))

    if min_dim == 1:
        # Every positive semidefinite matrix is separable when one of the local dimensions is 1.
        return True

    dim = [int(x) for x in dim]

    pt_state_alice = partial_trace(state, [1], dim)
    pt_state_bob = partial_trace(state, [0], dim)

    # Check the PPT criterion.
    if not is_ppt(state, 2, dim, tol):
        # Determined to be entangled via the PPT criterion.
        # A. Peres.
        # Separability criterion for density matrices.
        # Phys. Rev. Lett., 77:1413-1415, 1996.
        # Also, see Horodecki Theorem in https://arxiv.org/pdf/0811.2803.pdf.
        return False

    # Sometimes the PPT criterion is also sufficient for separability.
    if prod_dim <= 6 or min(dim) <= 1:
        # Determined to be separable via sufficiency of the PPT criterion in small dimensions
        # M. Horodecki, P. Horodecki, and R. Horodecki.
        # Separability of mixed states: Necessary and sufficient conditions.
        # Also, see Horodecki Theorem in https://arxiv.org/pdf/0811.2803.pdf.
        return is_ppt(state, 2, dim, tol)

    if (
        state_rank + np.linalg.matrix_rank(pt_state_alice)
        <= 2 * state.shape[0] * state.shape[1] - state.shape[0] - state.shape[1] + 2
        or state_rank + np.linalg.matrix_rank(pt_state_bob)
        <= 2 * state.shape[0] * state.shape[1] - state.shape[0] - state.shape[1] + 2
    ):
        # Determined to be separable via operational criterion of the PPT criterion for low-rank operators.
        # P. Horodecki, M. Lewenstein, G. Vidal, and I. Cirac.
        # Operational criterion and constructive checks for the separability of low-rank density matrices.
        # Phys. Rev. A, 62:032310, 2000.
        # TODO
        pass

    # Realignment (a.k.a computable cross-norm) criterion.
    if trace_norm(realignment(state, dim)) > 1 + tol:
        # Determined to be entangled via the realignment criterion.
        # K. Chen and L.-A. Wu.
        # A matrix realignment method for recognizing entanglement.
        # Quantum Inf. Comput., 3:193-202, 2003.
        return False

    # Another test that is strictly stronger than the realignment criterion.
    if trace_norm(realignment(state - np.kron(pt_state_alice, pt_state_bob), dim)) > np.sqrt(
        1 - np.trace(pt_state_alice ** 2 @ pt_state_bob ** 2)
    ):
        # Determined to be entangled by using Theorem 1 of reference.
        # C.-J. Zhang, Y.-S. Zhang, S. Zhang, and G.-C. Guo.
        # Entanglement detection beyond the cross-norm or realignment criterion.
        # Phys. Rev. A, 77:060301(R), 2008.
        return False

    # Obtain sorted list of eigenvalues in descending order.
    eig_vals, _ = np.linalg.eig(state)
    lam = eig_vals[np.argsort(-eig_vals)]

    # There are some separability tests that work specifically in the qubit-qudit (i.e., 2 \otimes n) case.
    # Check these tests.
    if min_dim == 2:
        # Check if X is separable from spectrum.
        if (lam[0] - lam[2 * max_dim - 1]) ** 2 <= 4 * lam[2 * max_dim - 2] * lam[
            2 * max_dim
        ] + tol ** 2:
            print("Determined to be separable by inspecting its eigenvalues.")
            print(
                "N. Johnston. Separability from spectrum for qubit-qudit states. Phys. Rev. A, 88:062330, 2013."
            )
            return True

    # For the rest of the block-matrix tests, we need the 2-dimensional subsystem to be the
    # first subsystem, so swap accordingly.
    # if dim[0] > 2:
    #    Xt = swap(state, [1, 2], dim)
    # else:
    #    Xt = state
    # commented out because pylint flagged this as an unused variable

    # Check the proximity of X with the maximally mixed state.
    if in_separable_ball(state):
        # Determined to be separable by closeness to the maximally mixed state.
        # L. Gurvits and H. Barnum. Largest separable balls around the maximally mixed bipartite quantum state.
        # Phys. Rev. A, 66:062311, 2002.
        return True

    # Check if X is a rank-1 perturbation of the identity, which is
    # necessarily separable if it's PPT, which we have already checked.
    if lam[1] - lam[prod_dim - 1] < tol ** 2:
        # Determined to be separable by being a small rank-1 perturbation of the maximally-mixed state.
        # G. Vidal and R. Tarrach. Robustness of entanglement.
        # Phys. Rev. A, 59:141-155, 1999.
        return True

    # The search for symmetric extensions.
    if any(has_symmetric_extension(state, level) for _ in range(2, level)):
        return True
    return False
