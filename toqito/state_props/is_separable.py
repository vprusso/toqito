"""Check if state is separable."""

from itertools import product

import numpy as np
from scipy.linalg import orth

from toqito.channel_ops.partial_channel import partial_channel
from toqito.channels import realignment
from toqito.channels.partial_trace import partial_trace
from toqito.matrix_props import is_positive_semidefinite, trace_norm
from toqito.perms.swap import swap
from toqito.perms.swap_operator import swap_operator
from toqito.state_props import in_separable_ball, is_ppt
from toqito.state_props.has_symmetric_extension import has_symmetric_extension
from toqito.state_props.schmidt_rank import schmidt_rank
from toqito.states.max_entangled import max_entangled


def is_separable(state: np.ndarray, dim: None | int | list[int] = None, level: int = 2, tol: float = 1e-8) -> bool:
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

    We provide the input as a density matrix :math:`\rho`.

    On the other hand, a random density matrix will be an entangled state (a separable state).

    >>> import numpy as np
    >>> from toqito.rand import random_density_matrix
    >>> from toqito.state_props import is_separable
    >>> rho_separable = np.array([[1, 0, 1, 0],
    ...                           [0, 1, 0, 1],
    ...                           [1, 0, 1, 0],
    ...                           [0, 1, 0, 1]])
    >>> is_separable(rho_separable)
    True

    >>> rho_not_separable = np.array([[ 0.13407875+0.j        , -0.08263926-0.17760437j,
    ...    -0.0135111 -0.12352182j,  0.0368423 -0.05563985j],
    ...   [-0.08263926+0.17760437j,  0.53338542+0.j        ,
    ...     0.19782968-0.04549732j,  0.11287093+0.17024249j],
    ...   [-0.0135111 +0.12352182j,  0.19782968+0.04549732j,
    ...     0.21254612+0.j        , -0.00875865+0.11144344j],
    ...   [ 0.0368423 +0.05563985j,  0.11287093-0.17024249j,
    ...    -0.00875865-0.11144344j,  0.11998971+0.j        ]])
    >>> is_separable(rho_not_separable)
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
        dim = np.array([dim, state_len / dim])
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
    is_ppt_state = is_ppt(state, 2, dim, tol)
    if not is_ppt_state:
        # Determined to be entangled via the PPT criterion. See (Peres_1996_Separability).
        # Also, see Horodecki Theorem in (Gühne_2009_Horodecki).
        return False

    # Sometimes the PPT criterion is also sufficient for separability.
    elif prod_dim <= 6 or min(dim) <= 1:
        # Determined to be separable via sufficiency of the PPT criterion in small dimensions.
        # See (Horodecki_1996_PPT_small_dimensions).
        # Also, see Horodecki Theorem in (Gühne_2009_Horodecki).
        return is_ppt_state

    if (
        state_rank + np.linalg.matrix_rank(pt_state_alice)
        <= 2 * state.shape[0] * state.shape[1] - state.shape[0] - state.shape[1] + 2
        or state_rank + np.linalg.matrix_rank(pt_state_bob)
        <= 2 * state.shape[0] * state.shape[1] - state.shape[0] - state.shape[1] + 2
    ):
        # Determined to be separable via operational criterion of the PPT criterion for low-rank operators.
        # See (Horodecki_2000_PPT_low_rank).
        # TODO
        pass

    # Realignment (a.k.a computable cross-norm) criterion.
    if trace_norm(realignment(state, dim)) > 1 + tol:
        # Determined to be entangled via the realignment criterion. See (Chen_2003_Realignment).
        return False

    # Another test that is strictly stronger than the realignment criterion.
    if trace_norm(realignment(state - np.kron(pt_state_alice, pt_state_bob), dim)) > np.sqrt(
        1 - np.trace(pt_state_alice**2 @ pt_state_bob**2)
    ):
        # Determined to be entangled by using Theorem 1 of (Zhang_2008_Beyond_realignment).
        return False

    # Obtain sorted list of eigenvalues in descending order.
    eig_vals, _ = np.linalg.eig(state)
    lam = eig_vals[np.argsort(-eig_vals)]

    # There are some separability tests that work specifically in the qubit-qudit (i.e., 2 \otimes n) case.
    # Check these tests.
    if min_dim == 2:
        # Check if X is separable from spectrum.
        # Determined to be separable by inspecting its eigenvalues. See (Johnston_2013_Spectrum).
        if (lam[0] - lam[2 * max_dim - 2]) ** 2 <= 4 * lam[2 * max_dim - 3] * lam[2 * max_dim - 1] + tol**2:
            return True

        # For the rest of the block matrix tests, we need the 2-dimensional
        # subsystem to be the *first* subsystem, so swap accordingly.
        state_t = swap(state, [1, 2], dim) if dim[0] > 2 else state

        # Check if Lemma 1 of (Johnston_2013_Spectrum) applies to X. Also check the Hildebrand 2xn results.
        A = state_t[:max_dim, :max_dim]
        B = state_t[:max_dim, max_dim : 2 * max_dim]
        C = state_t[max_dim : 2 * max_dim, max_dim : 2 * max_dim]

        # Determined to be separable by being a perturbed block Hankel matrix of (Hildebrand_2005_Cone).
        if np.linalg.matrix_rank(B - B.conj().T) <= 1 and is_ppt_state:
            return True

        X_2n_ppt_check = np.vstack((np.hstack(((5 / 6) * A - C / 6, B)), np.hstack((B.conj().T, (5 / 6) * C - A / 6))))
        # Determined to be separable via the homothetic images approach of (Hildebrand_2005_Cone).
        if is_positive_semidefinite(X_2n_ppt_check) and is_ppt(X_2n_ppt_check, 2, [2, max_dim]):
            return True

        # Determined to be separable by using Lemma 1 of (Johnston_2013_Spectrum).
        if np.linalg.norm(B) ** 2 <= np.min(np.real(np.linalg.eig(A))) * np.min(np.real(np.linalg.eig(C))) + tol**2:
            return True

    # There are conditions that are both necessary and sufficient when both
    # dimensions are 3 and the rank is 4
    if state_rank == 4 and min_dim == 3 and max_dim == 3:
        # This method computes more determinants than are actually
        # necessary, but the speed loss isn't too great
        p = np.zeros((6, 7, 8, 9))  # initialize.
        q = orth(state)

        # This loop does not access all positions in `p`, and that's expected.
        for j, k, n, m in product(range(6, 0, -1), range(7, 0, -1), range(8, 0, -1), range(9, 0, -1)):
            if j < k < n < m:
                p[j - 1, k - 1, n - 1, m - 1] = np.linalg.det(q[[j - 1, k - 1, n - 1, m - 1], :])

        F = np.linalg.det(
            np.array(
                [
                    [
                        p[0, 1, 3, 4],
                        p[0, 2, 3, 5],
                        p[1, 2, 4, 5],
                        p[0, 1, 3, 5] + p[0, 2, 3, 4],
                        p[0, 1, 4, 5] + p[1, 2, 3, 4],
                        p[0, 2, 4, 5] + p[1, 2, 3, 5],
                    ],
                    [
                        p[0, 1, 6, 7],
                        p[0, 2, 6, 8],
                        p[1, 2, 7, 8],
                        p[0, 1, 6, 8] + p[0, 2, 6, 7],
                        p[0, 1, 7, 8] + p[1, 2, 6, 7],
                        p[0, 2, 7, 8] + p[1, 2, 6, 8],
                    ],
                    [
                        p[3, 4, 6, 7],
                        p[3, 5, 6, 8],
                        p[4, 5, 7, 8],
                        p[3, 4, 6, 8] + p[3, 5, 6, 7],
                        p[3, 4, 7, 8] + p[4, 5, 6, 7],
                        p[3, 5, 7, 8] + p[4, 5, 6, 8],
                    ],
                    [
                        p[0, 1, 3, 7] - p[0, 1, 4, 6],
                        p[0, 2, 3, 8] - p[0, 2, 5, 6],
                        p[1, 2, 4, 8] - p[1, 2, 5, 7],
                        p[0, 1, 3, 8] - p[0, 1, 5, 6] + p[0, 2, 3, 7] - p[0, 2, 4, 6],
                        p[0, 1, 4, 8] - p[0, 1, 5, 7] + p[1, 2, 3, 7] - p[1, 2, 4, 6],
                        p[0, 2, 4, 8] - p[0, 2, 5, 7] + p[1, 2, 3, 8] - p[1, 2, 5, 6],
                    ],
                    [
                        p[0, 3, 4, 7] - p[1, 3, 4, 6],
                        p[0, 3, 5, 8] - p[2, 3, 5, 6],
                        p[1, 4, 5, 8] - p[2, 4, 5, 7],
                        p[0, 3, 4, 8] - p[1, 3, 5, 6] + p[0, 3, 5, 7] - p[2, 3, 4, 6],
                        p[0, 4, 5, 7] - p[1, 4, 5, 6] + p[1, 3, 4, 8] - p[2, 3, 4, 7],
                        p[0, 4, 5, 8] - p[2, 3, 5, 7] + p[1, 3, 5, 8] - p[2, 4, 5, 6],
                    ],
                    [
                        p[0, 4, 6, 7] - p[1, 3, 6, 7],
                        p[0, 5, 6, 8] - p[2, 3, 6, 8],
                        p[1, 5, 7, 8] - p[2, 4, 7, 8],
                        p[0, 4, 6, 8] - p[1, 3, 6, 8] + p[0, 5, 6, 7] - p[2, 3, 6, 7],
                        p[0, 4, 7, 8] - p[1, 3, 7, 8] + p[1, 5, 6, 7] - p[2, 4, 6, 7],
                        p[0, 5, 7, 8] - p[2, 3, 7, 8] + p[1, 5, 6, 8] - p[2, 4, 6, 8],
                    ],
                ]
            )
        )

        # Matrix is separable iff F is zero (or suffiently close to it, for numerical reasons)
        return abs(F) < max(tol**2, eps ** (3 / 4))

    # Check the proximity of X with the maximally mixed state.
    if in_separable_ball(state):
        # Determined to be separable by closeness to the maximally mixed state. See (Gurvits_2002_Ball).
        return True

    # Check if X is a rank-1 perturbation of the identity, which is
    # necessarily separable if it's PPT, which we have already checked.
    if lam[1] - lam[prod_dim - 1] < tol**2:
        # Determined to be separable by being a small rank-1 perturbation of the maximally-mixed state.
        # See (Vidal_1999_Robust).
        return True

    # Determined to be separable by having operator Schmidt rank at most 2. See (Cariello_2013_Weak_irreducible).
    if schmidt_rank(state, dim) <= 2:
        return True

    # There is a family of known optimal positive maps in the qutrit-qutrit
    # case. Check for entanglement using these.
    if dim[0] == 3 and dim[1] == 3:
        phi = max_entangled(3, False, False)
        for t in np.arange(0, 1.0, 0.1):
            t_ = t  # This is to evade ruff `PLW2901` error.
            for j in range(2):
                # This is a weird way of using both t and 1/t as indices for the maps Phi we generate.
                if t_ > 0:
                    t_ = 1 / t_
                elif j > 0:
                    break

                a = (1 - t_) ** 2 / (1 - t_ + t_**2)
                b = t_**2 / (1 - t_ + t_**2)
                c = 1 / (1 - t_ + t_**2)
                Phi = np.diag([a + 1, c, b, b, a + 1, c, c, b, a + 1]) - phi @ phi.conj().T

                # See (Ha_2011_Positive_map).
                if not is_positive_semidefinite(partial_channel(state, Phi, 2, dim)):
                    return False

    # Use the Breuer-Hall positive maps (in even dimensions only) based on
    # antisymmetric unitary matrices.
    for p in range(2):
        if np.remainder(dim[p], 2) == 0:
            phi = max_entangled(dim[p], False, False)
            U = np.kron(
                np.eye(dim[p]), np.fliplr(np.diag(np.array([[np.ones((dim[p] / 2, 1))], [-np.ones(dim(p) / 2, 1)]])))
            )
            Phi = np.diag(np.ones((dim[p] ** 2, 1))) - phi @ phi.conj().T - U @ swap_operator(dim[p]) @ U.conj().T

            # Determined to be entangled via the Breuer-Hall positive maps based on antisymmetric unitary matrices.
            # See (Breuer_2006_Mixed) and (Hall_2006_Indecomposable).
            if not is_positive_semidefinite(partial_channel(state, Phi, p + 1, dim)):
                return False

    # The search for symmetric extensions.
    if any(has_symmetric_extension(state, level) for _ in range(1, level)):
        return True
    return False
