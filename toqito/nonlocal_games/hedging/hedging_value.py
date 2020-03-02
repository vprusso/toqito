"""Semidefinite programs for obtaining values of quantum hedging scenarios."""
import cvxpy
import numpy as np

from toqito.perms.permutation_operator import permutation_operator
from toqito.super_operators.partial_trace import partial_trace_cvx


class HedgingValue:
    def __init__(self, q_a: np.ndarray, num_reps: int) -> None:
        """
        Initialize the variables for semidefinite program.

        :param q_a:
        :param num_reps:
        """
        self._q_a = q_a
        self._num_reps = num_reps

        self._sys = list(range(1, 2*self._num_reps, 2))
        if len(self._sys) == 1:
            self._sys = self._sys[0]

        self._dim = 2 * np.ones((1, 2*self._num_reps)).astype(int).flatten()
        self._dim = self._dim.tolist()

        # For the dual problem, the following unitary operator is used to
        # permute the subsystems of Alice and Bob which is defined by the
        # action:
        #   π(y1 ⊗ y2 ⊗ x1 ⊗ x2) = y1 ⊗ x1 ⊗ y2 ⊗ x2
        # for all y1 ∈ Y1, y2 ∈ Y2, x1 ∈ X1, x2 ∈ X2.).
        l_1 = list(range(1, self._num_reps + 1))
        l_2 = list(range(self._num_reps + 1, self._num_reps ** 2 + 1))
        if self._num_reps == 1:
            perm = [1]
        else:
            perm = [*sum(zip(l_1, l_2), ())]
        self._pperm = permutation_operator(self._num_reps, perm)

    def max_prob_outcome_a_primal(self) -> float:
        """
        Compute the maximal probability for calculating outcome "a".

        The primal problem for the maximal probability of "a" is given as:

        ..math::
        `
        `

        :return: The optimal maximal probability for obtaining outcome "a".
        """
        x_var = cvxpy.Variable((4**self._num_reps, 4**self._num_reps), PSD=True)
        objective = cvxpy.Maximize(cvxpy.trace(self._q_a.conj().T @ x_var))
        constraints = [
            partial_trace_cvx(x_var,
                              self._sys,
                              self._dim) == np.identity(2 ** self._num_reps)
        ]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()

    def max_prob_outcome_a_dual(self) -> float:
        """
        Compute the maximal probability for calculating outcome "a".

        The dual problem for the maximal probability of "a" is given as:

        ..math::
        `
        `

        :return: The optimal maximal probability for obtaining outcome "a".
        """
        y_var = cvxpy.Variable((2**self._num_reps,
                                2**self._num_reps), hermitian=True)
        objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

        kron_var = cvxpy.kron(np.eye(2**self._num_reps), y_var)
        if self._num_reps == 1:
            u_var = cvxpy.multiply(cvxpy.multiply(self._pperm,
                                                  kron_var),
                                   self._pperm.conj().T)
            constraints = [cvxpy.real(u_var) >> self._q_a]
        else:
            constraints = [
                cvxpy.real(self._pperm @
                           kron_var @
                           self._pperm.conj().T) >> self._q_a]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()

    def min_prob_outcome_a_primal(self) -> float:
        """
        Compute the minimal probability for calculating outcome "a".

        The primal problem for the minimal probability of "a" is given as:

        ..math::
        `
        `

        :return: The optimal minimal probability for obtaining outcome "a".
        """
        x_var = cvxpy.Variable((4**self._num_reps, 4**self._num_reps), PSD=True)
        objective = cvxpy.Minimize(cvxpy.trace(self._q_a.conj().T @ x_var))
        constraints = [
            partial_trace_cvx(x_var,
                              self._sys,
                              self._dim) == np.identity(2 ** self._num_reps)
        ]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()

    def min_prob_outcome_a_dual(self) -> float:
        """
        Compute the minimal probability for calculating outcome "a".

        The dual problem for the minimal probability of "a" is given as:

        ..math::
        `
        `

        :return: The optimal minimal probability for obtaining outcome "a".
        """
        y_var = cvxpy.Variable((2**self._num_reps,
                                2**self._num_reps), hermitian=True)
        objective = cvxpy.Maximize(cvxpy.trace(cvxpy.real(y_var)))

        kron_var = cvxpy.kron(np.eye(2**self._num_reps), y_var)

        if self._num_reps == 1:
            u_var = cvxpy.multiply(cvxpy.multiply(self._pperm, kron_var),
                                   self._pperm.conj().T)
            constraints = [cvxpy.real(u_var) << self._q_a]
        else:
            constraints = [cvxpy.real(self._pperm @
                                      kron_var @
                                      self._pperm.conj().T) << self._q_a]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()
