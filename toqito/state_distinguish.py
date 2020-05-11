"""Quantum state distinguishability scenarios and protocols."""
from typing import List

import cvxpy
import numpy as np

from toqito.channels import partial_transpose


class StateDistinguish:
    r"""
    Class to distinguish ensembles of quantum states.

    Test.
    """

    def __init__(self, states: List[np.ndarray], probs: List[float] = None):
        """Create distinguishability scenarios.

        :param states: A list of density operators (matrices) corresponding to
                       quantum states.
        :param probs: A list of probabilities where `probs[i]` corresponds to
                      the probability that `states[i]` is selected by Alice.
        """
        # Assume that at least one state is provided.
        if states is None or states == []:
            raise ValueError(
                "InvalidStates: There must be at least one state provided."
            )

        # Assume uniform probability if no specific distribution is given.
        if probs is None:
            probs = [1 / len(states)] * len(states)
        if not np.isclose(sum(probs), 1):
            raise ValueError("Invalid: Probabilities must sum to 1.")

        _, dim_y = states[0].shape

        # The variable `states` is provided as a list of vectors. Transform them
        # into density matrices.
        if dim_y == 1:
            for i, state_ket in enumerate(states):
                states[i] = state_ket * state_ket.conj().T

        self._states = states
        self._probs = probs

    def state_distinguishability(self) -> float:
        r"""
        Compute probability of state distinguishability [ELD03]_.

        The "quantum state distinguishability" problem involves a collection of
        :math:`n` quantum states

        .. math::
            \rho = \{ \rho_0, \ldots, \rho_n \},

        as well as a list of corresponding probabilities

        .. math::
            p = \{ p_0, \ldots, p_n \}

        Alice chooses :math:`i` with probability :math:`p_i` and creates the state
        :math:`rho_i`

        Bob wants to guess which state he was given from the collection of states.

        This function implements the following semidefinite program that provides
        the optimal probability with which Bob can conduct quantum state
        distinguishability.

        .. math::

            \begin{align*}
                \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i,
                \rho_i \rangle \\
                \text{subject to:} \quad & M_0 + \ldots + M_n = \mathbb{I},\\
                                         & M_0, \ldots, M_n \geq 0
            \end{align*}

        Examples
        ==========

        State distinguishability for two state density matrices.

        >>> from toqito.states import basis, bell
        >>> from toqito.state_distinguish import StateDistinguish
        >>> e_0, e_1 = basis(2, 0), basis(2, 1)
        >>> e_00 = e_0 * e_0.conj().T
        >>> e_11 = e_1 * e_1.conj().T
        >>> states = [e_00, e_11]
        >>> probs = [1 / 2, 1 / 2]
        >>> s_d = StateDistinguish(states, probs)
        >>> res = s_d.state_distinguishability(states, probs)
        0.5000000000006083

        References
        ==========
        .. [ELD03] Eldar, Yonina C.
            "A semidefinite programming approach to optimal unambiguous
            discrimination of quantum states."
            IEEE Transactions on information theory 49.2 (2003): 446-456.
            https://arxiv.org/abs/quant-ph/0206093


        :return: The optimal probability with which Bob can distinguish the state.
        """
        obj_func = []
        measurements = []
        constraints = []

        dim_x, _ = self._states[0].shape
        for i, _ in enumerate(self._states):
            measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

            obj_func.append(
                self._probs[i] * cvxpy.trace(self._states[i].conj().T @ measurements[i])
            )

        constraints.append(sum(measurements) == np.identity(dim_x))

        objective = cvxpy.Maximize(sum(obj_func))
        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve()

        return 1 / len(self._states) * sol_default

    def ppt_distinguishability(self) -> float:
        r"""
        Compute probability of distinguishing a state via PPT measurements [COS13]_.

        Implements the semidefinite program (SDP) whose optimal value is equal to
        the maximum probability of perfectly distinguishing orthogonal maximally
        entangled states using any PPT measurement; a measurement whose operators
        are positive under partial transpose. This SDP was explicitly provided in
        [COS13]_.

        Specifically, the function implements the dual problem (as this is
        computationally more efficient) and is defined as:

        .. math::

            \begin{equation}
                \begin{aligned}
                    \text{minimize:} \quad & \frac{1}{k} \text{Tr}(Y) \\
                    \text{subject to:} \quad & Y \geq \text{T}_{\mathcal{A}}
                                              (\rho_j), \quad j = 1, \ldots, k, \\
                                             & Y \in \text{Herm}(\mathcal{A} \otimes
                                              \mathcal{B}).
                \end{aligned}
            \end{equation}

        Examples
        ==========

        Consider the following Bell states

        .. math::
            \begin{equation}
                \begin{aligned}
                |\psi_0 \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, &\quad
                |\psi_1 \rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}, \\
                |\psi_2 \rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}, &\quad
                |\psi_3 \rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}.
                \end{aligned}
            \end{equation}

        It was illustrated in [YDY12]_ that for the following set of states:

        The PPT distinguishability of the following states

        .. math::
            \begin{equation}
                \rho_1^{(2)} = \psi_0 \otimes \psi_0, \quad
                \rho_2^{(2)} = \psi_1 \otimes \psi_1, \quad
            \end{equation}

        should yield :math:`7/8 ~ 0.875` as was proved in [YDY12]_.

        >>> from toqito.states import bell
        >>> from toqito.state_distinguish import StateDistinguish
        >>> # Bell vectors:
        >>> psi_0 = bell(0)
        >>> psi_1 = bell(2)
        >>> psi_2 = bell(3)
        >>> psi_3 = bell(1)
        >>>
        >>> # YYD vectors from [YDY12]_.
        >>> x_1 = np.kron(psi_0, psi_0)
        >>> x_2 = np.kron(psi_1, psi_3)
        >>> x_3 = np.kron(psi_2, psi_3)
        >>> x_4 = np.kron(psi_3, psi_3)
        >>>
        >>> # YYD density matrices.
        >>> rho_1 = x_1 * x_1.conj().T
        >>> rho_2 = x_2 * x_2.conj().T
        >>> rho_3 = x_3 * x_3.conj().T
        >>> rho_4 = x_4 * x_4.conj().T
        >>>
        >>> states = [rho_1, rho_2, rho_3, rho_4]
        >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        >>> ydy = StateDistinguish(states, probs)
        >>> ydy.ppt_distinguishability()
        0.875

        References
        ==========
        .. [COS13] Cosentino, Alessandro.
            "Positive-partial-transpose-indistinguishable states via semidefinite
            programming."
            Physical Review A 87.1 (2013): 012321.
            https://arxiv.org/abs/1205.1031

        .. [YDY12] Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
            "Four locally indistinguishable ququad-ququad orthogonal
            maximally entangled states."
            Physical review letters 109.2 (2012): 020506.
            https://arxiv.org/abs/1107.3224

        :return: The optimal probability with which the states can be distinguished
                 via PPT measurements.
        """
        constraints = []
        dim_x, _ = self._states[0].shape
        y_var = cvxpy.Variable((dim_x, dim_x), hermitian=True)
        objective = (
            1 / len(self._states) * cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))
        )

        dim = int(np.log2(dim_x))
        dim_list = [2] * int(np.log2(dim_x))
        sys_list = list(range(1, dim, 2))

        for i, _ in enumerate(self._states):
            constraints.append(
                cvxpy.real(y_var)
                >> partial_transpose(self._states[i], sys=sys_list, dim=dim_list)
            )

        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve()

        return sol_default

    def conclusive_state_exclusion(self) -> float:
        r"""
        Compute probability of conclusive single state exclusion.

        The "quantum state exclusion" problem involves a collection of :math:`n`
        quantum states

        .. math::
            \rho = \{ \rho_0, \ldots, \rho_n \},

        as well as a list of corresponding probabilities

        .. math::
            p = \{ p_0, \ldots, p_n \}

        Alice chooses :math:`i` with probability :math:`p_i` and creates the state
        :math:`\rho_i`.

        Bob wants to guess which state he was *not* given from the collection of
        states. State exclusion implies that ability to discard (with certainty) at
        least one out of the "n" possible quantum states by applying a measurement.

        This function implements the following semidefinite program that provides
        the optimal probability with which Bob can conduct quantum state exclusion.

            .. math::
                \begin{equation}
                    \begin{aligned}
                        \text{minimize:} \quad & \sum_{i=0}^n p_i \langle M_i,
                                                    \rho_i \rangle \\
                        \text{subject to:} \quad & M_0 + \ldots + M_n =
                                                   \mathbb{I}, \\
                                                 & M_0, \ldots, M_n >= 0
                    \end{aligned}
                \end{equation}

        The conclusive state exclusion SDP is written explicitly in [BJOP14]_. The
        problem of conclusive state exclusion was also thought about under a
        different guise in [PBR12]_.

        Examples
        ==========

        Consider the following two Bell states

        .. math::
            u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \\
            u_1 = \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).

        For the corresponding density matrices :math:`\rho_0 = u_0 u_0^*` and
        :math:`\rho_1 = u_1 u_1^*`, we may construct a set

        .. math::
            \rho = \{\rho_0, \rho_1 \}

        such that

        .. math::
            p = \{1/2, 1/2\}.

        It is not possible to conclusively exclude either of the two states. We can
        see that the result of the function in `toqito` yields a value of :math`0`
        as the probability for this to occur.

        >>> from toqito.state_distinguish import StateDistinguish
        >>> from toqito.states import bell
        >>> import numpy as np
        >>> rho1 = bell(0) * bell(0).conj().T
        >>> rho2 = bell(1) * bell(1).conj().T
        >>>
        >>> states = [rho1, rho2]
        >>> probs = [1/2, 1/2]
        >>>
        >>> s_d = StateDistinguish(states, probs)
        >>> s_d.conclusive_state_exclusion()
        1.6824720366950206e-09

        References
        ==========
        .. [PBR12] "On the reality of the quantum state"
            Pusey, Matthew F., Jonathan Barrett, and Terry Rudolph.
            Nature Physics 8.6 (2012): 475-478.
            arXiv:1111.3328

        .. [BJOP14] "Conclusive exclusion of quantum states"
            Somshubhro Bandyopadhyay, Rahul Jain, Jonathan Oppenheim,
            Christopher Perry
            Physical Review A 89.2 (2014): 022336.
            arXiv:1306.4683

        :return: The optimal probability with which Bob can guess the state he was
                 not given from `states`.
        """
        obj_func = []
        measurements = []
        constraints = []
        dim_x, _ = self._states[0].shape

        for i, _ in enumerate(self._states):
            measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

            obj_func.append(
                self._probs[i] * cvxpy.trace(self._states[i].conj().T @ measurements[i])
            )

        constraints.append(sum(measurements) == np.identity(dim_x))

        if np.iscomplexobj(self._states[0]):
            objective = cvxpy.Minimize(cvxpy.real(sum(obj_func)))
        else:
            objective = cvxpy.Minimize(sum(obj_func))

        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve()

        return 1 / len(self._states) * sol_default

    def unambiguous_state_exclusion(self) -> float:
        r"""
        Compute probability of unambiguous state exclusion [BJOPUS14]_.

        This function implements the following semidefinite program that provides
        the optimal probability with which Bob can conduct quantum state exclusion.

        .. math::

            \begin{align*}
                \text{maximize:} \quad & \sum_{i=0}^n \sum_{j=0}^n
                                         \langle M_i, \rho_j \rangle \\
                \text{subject to:} \quad & \sum_{i=0}^n M_i \leq \mathbb{I},\\
                                         & \text{Tr}(\rho_i M_i) = 0,
                                           \quad \quad \forall 1  \leq i \leq n, \\
                                         & M_0, \ldots, M_n \geq 0
            \end{align*}

        Examples
        ==========

        Consider the following two Bell states

        .. math::
            u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \\
            u_1 = \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).

        For the corresponding density matrices :math:`\rho_0 = u_0 u_0^*` and
        :math:`\rho_1 = u_1 u_1^*`, we may construct a set

        .. math::
            \rho = \{\rho_0, \rho_1 \}

        such that

        .. math::
            p = \{1/2, 1/2\}.

        It is not possible to unambiguously exclude either of the two states. We can
        see that the result of the function in `toqito` yields a value of :math:`0`
        as the probability for this to occur.

        >>> from toqito.state_distinguish import StateDistinguish
        >>> from toqito.states import bell
        >>> import numpy as np
        >>> rho1 = bell(0) * bell(0).conj().T
        >>> rho2 = bell(1) * bell(1).conj().T
        >>>
        >>> states = [rho1, rho2]
        >>> probs = [1/2, 1/2]
        >>>
        >>> s_d = StateDistinguish(states, probs)
        >>> s_d.unambiguous_state_exclusion()
        -7.250173600116328e-18

        References
        ==========
        .. [BJOPUS14] "Conclusive exclusion of quantum states"
            Somshubhro Bandyopadhyay, Rahul Jain, Jonathan Oppenheim,
            Christopher Perry
            Physical Review A 89.2 (2014): 022336.
            arXiv:1306.4683

        :return: The optimal probability with which Bob can guess the state he was
                 not given from `states` with certainty.
        """
        obj_func = []
        measurements = []
        constraints = []
        dim_x, _ = self._states[0].shape

        for i, _ in enumerate(self._states):
            measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

            obj_func.append(
                self._probs[i] * cvxpy.trace(self._states[i].conj().T @ measurements[i])
            )

            constraints.append(cvxpy.trace(self._states[i] @ measurements[i]) == 0)

        constraints.append(sum(measurements) <= np.identity(dim_x))

        if np.iscomplexobj(self._states[0]):
            objective = cvxpy.Maximize(cvxpy.real(sum(obj_func)))
        else:
            objective = cvxpy.Maximize(sum(obj_func))

        problem = cvxpy.Problem(objective, constraints)
        sol_default = problem.solve()

        return 1 / len(self._states) * sol_default
