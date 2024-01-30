"""PPT symmetric extension hierarchy."""


import cvxpy
import numpy as np

from toqito.channels import partial_trace, partial_transpose
from toqito.perms import symmetric_projection

from .state_helper import __is_probs_valid, __is_states_valid


def symmetric_extension_hierarchy(
    states: list[np.ndarray],
    probs: list[float] = None,
    level: int = 2,
    dim: int | list[int] = None,
) -> float:
    r"""Compute optimal value of the symmetric extension hierarchy SDP :cite:`Navascues_2008_Pure`.

    The probability of distinguishing a given set of states via PPT measurements serves as a natural
    upper bound to the value of obtaining via separable measurements. Due to the nature of separable
    measurements, it is not possible to optimize directly over these objects via semidefinite
    programming techniques.

    We can, however, construct a hierarchy of semidefinite programs that attains closer and closer
    approximations at the separable value via the techniques described in :cite:`Navascues_2008_Pure`.

    The mathematical form of this hierarchy implemented here is explicitly given from equation 4.55
    in :cite:`Cosentino_2015_QuantumState`.

    .. math::

        \begin{equation}
            \begin{aligned}
                \text{maximize:} \quad & \sum_{k=1}^N p_k \langle \rho_k, \mu(k) \rangle, \\
                \text{subject to:} \quad & \sum_{k=1}^N \mu(k) =
                                           \mathbb{I}_{\mathcal{X} \otimes \mathcal{Y}}, \\
                                        & \text{Tr}_{\mathcal{Y}_2 \otimes \ldots \otimes
                                          \mathcal{Y}_s}(X_k) = \mu(k), \\
                                        & \left( \mathbb{I}_{\mathcal{X}} \otimes
                                          \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ovee \ldots \ovee
                                          \mathcal{Y}_s} \right) X_k
                                          \left(\mathbb{I}_{\mathcal{X}} \otimes
                                          \Pi_{\mathcal{Y} \ovee \mathcal{Y}_2 \ovee \ldots \ovee
                                          \mathcal{Y}_s} \right)
                                          = X_k \\
                                        & \text{T}_{\mathcal{X}}(X_k) \in \text{Pos}\left(
                                            \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                            \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                        & \text{T}_{\mathcal{Y}_2 \otimes \ldots \otimes
                                            \mathcal{Y}_s}(X_k) \in \text{Pos}\left(
                                            \mathcal{X} \otimes \mathcal{Y} \otimes \mathcal{Y}_2
                                            \otimes \ldots \otimes \mathcal{Y}_s \right), \\
                                        & X_1, \ldots, X_N \in
                                          \text{Pos}\left(\mathcal{X} \otimes \mathcal{Y} \otimes
                                          \mathcal{Y}_2 \otimes \ldots \otimes \mathcal{Y}_s
                                          \right).
            \end{aligned}
        \end{equation}

    Examples
    ==========

    It is known from :cite:`Cosentino_2015_QuantumState` that distinguishing three Bell states along with a resource
    state :math:`|\tau_{\epsilon}\rangle` via separable measurements has the following closed form

    .. math::
        \frac{1}{3} \left(2 + \sqrt{1 - \epsilon^2} \right)

    where the resource state is defined as

    .. math::
        |\tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} |00\rangle +
                                   \sqrt{\frac{1-\epsilon}{2}} |11\rangle.

    The value of optimally distinguishing these states via PPT measurements is strictly larger than
    the value one obtains from separable measurements. Calculating the first level of the hierarchy
    provides for us the optimal value of PPT measurements.

    Consider a fixed value of :math:`\epsilon = 0.5`.

    >>> from toqito.states import basis, bell
    >>> from toqito.perms import swap
    >>> import numpy as np
    >>>
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)
    >>>
    >>> # Define the resource state.
    >>> eps = 0.5
    >>> eps_state = np.sqrt((1+eps)/2) * e_00 + np.sqrt((1-eps)/2) * e_11
    >>> eps_dm = eps_state * eps_state.conj().T
    >>>
    >>> # Define the ensemble of states to be distinguished.
    >>> states = [
    ...     np.kron(bell(0) * bell(0).conj().T, eps_dm),
    ...     np.kron(bell(1) * bell(1).conj().T, eps_dm),
    ...     np.kron(bell(2) * bell(2).conj().T, eps_dm),
    ...     np.kron(bell(3) * bell(3).conj().T, eps_dm),
    ... ]
    >>>
    >>> # Ensure the distinguishability is conducted on the proper spaces.
    >>> states = [
    ...     swap(states[0], [2, 3], [2, 2, 2, 2]),
    ...     swap(states[1], [2, 3], [2, 2, 2, 2]),
    ...     swap(states[2], [2, 3], [2, 2, 2, 2]),
    ... ]
    >>>
    >>> # Calculate the first level of the symmetric extension hierarchy. This
    >>> # is simply the value of optimally distinguishing via PPT measurements.
    >>> symmetric_extension_hierarchy(states=states, probs=None, level=1)
    0.9915817434994775
    >>>
    >>> # Calculating the second value gets closer to the separable value.
    >>> symmetric_extension_hierarchy(states=states, probs=None, level=2)
    0.958305796189204
    >>>
    >>> # As proven in :cite:`Cosentino_2015_QuantumState`, the true separable value of distinguishing the
    >>> # three Bell states is:
    >>> 1/3 * (2 + np.sqrt(1 - eps**2))
    0.9553418012614794
    >>>
    >>> # Computing further levels of the hierarchy would eventually converge to
    >>> # this value, however, the higher the level, the more computationally
    >>> # demanding the SDP becomes.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param states: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param level: Level of the hierarchy to compute.
    :param dim: The default has both subsystems of equal dimension.
    :return: The optimal probability of the symmetric extension hierarchy SDP for level
            :code:`level`.

    """
    obj_func = []
    meas = []
    x_var = []
    constraints = []

    __is_states_valid(states)
    if probs is None:
        probs = [1 / len(states)] * len(states)
    __is_probs_valid(probs)

    dim_xy, n_cols = states[0].shape

    # The variable `states` is provided as a list of vectors. Transform them
    # into density matrices.
    if n_cols == 1:
        for i, state_ket in enumerate(states):
            states[i] = state_ket * state_ket.conj().T

    # Set default dimension if none was provided.
    if dim is None:
        dim = int(np.round(np.sqrt(dim_xy)))

    # Allow the user to enter in a single integer for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim_xy / dim])  # pylint: disable=redefined-variable-type
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * dim_xy * np.finfo(float).eps:
            raise ValueError("If `dim` is a scalar, it must evenly divide the length of the state.")
        dim[1] = int(np.round(dim[1]))

    dim_x, dim_y = int(dim[0]), int(dim[1])

    dim_list = [dim_x] + [dim_y] * level
    dim_list = np.int_(dim_list)  # pylint: disable=redefined-variable-type
    # The `sys_list` variable contains the numbering pertaining to the symmetrically extended
    # spaces.
    sys_list = list(range(2, 2 + level - 1))
    sym = symmetric_projection(dim_y, level)

    dim_xyy = np.prod(dim_list)
    for k, item in enumerate(states):
        meas.append(cvxpy.Variable((dim_xy, dim_xy), PSD=True))
        x_var.append(cvxpy.Variable((dim_xyy, dim_xyy), PSD=True))
        constraints.append(partial_trace(x_var[k], sys_list, dim_list) == meas[k])
        constraints.append(
            np.kron(np.identity(dim_x), sym) @ x_var[k] @ np.kron(np.identity(dim_x), sym)
            == x_var[k]
        )
        constraints.append(partial_transpose(x_var[k], [0], dim_list) >> 0)
        for sys in range(level - 1):
            constraints.append(partial_transpose(x_var[k], [sys + 2], dim_list) >> 0)

        obj_func.append(probs[k] * cvxpy.trace(item.conj().T @ meas[k]))

    constraints.append(sum(meas) == np.identity(dim_xy))

    objective = cvxpy.Maximize(sum(obj_func))
    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default
