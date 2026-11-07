"""Two-player XOR game."""

from functools import cached_property

import cvxpy
import numpy as np

from toqito._sdp import solve_sdp
from toqito.nonlocal_games.nonlocal_game import NonlocalGame, QuantumStrategy


def _clifford_generators(count: int) -> np.ndarray:
    r"""Return ``count`` pairwise anticommuting Hermitian involutions on \(2^{\lceil count/2 \rceil}\) dimensions.

    The Jordan-Wigner construction on \(q\) qubits gives \(2q\) of them: \(Z^{\otimes (j-1)} \otimes X \otimes
    \mathbb{1}\) and \(Z^{\otimes (j-1)} \otimes Y \otimes \mathbb{1}\) for \(j = 1, \ldots, q\).
    """
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    num_qubits = max(1, -(-count // 2))
    generators = []
    for qubit in range(num_qubits):
        for pauli in (pauli_x, pauli_y):
            factors = [pauli_z] * qubit + [pauli] + [np.eye(2)] * (num_qubits - qubit - 1)
            operator = factors[0]
            for factor in factors[1:]:
                operator = np.kron(operator, factor)
            generators.append(operator)
    return np.array(generators[:count])


class XORGame:
    r"""Create two-player XOR game object.

    Calculates the optimal probability that Alice and Bob win the game if they
    are allowed to determine a join strategy beforehand, but not allowed to
    communicate during the game itself.

    The quantum value of an XOR game can be solved via the semidefinite program
    from [@cleve2010consequences].

    This function is adapted from the QETLAB package [@qetlablink].

    A tutorial is available in the documentation. Go to
    [Calculating the quantum and classical value of a two-player XOR game]
    (../../../generated/gallery/nonlocal_games/xor_quantum_val.md).

    Examples:
        The CHSH game

        The CHSH game is a two-player nonlocal game with the following probability
        distribution and question and answer sets [@cleve2008strong].

        \[
            \begin{equation}
                    \begin{aligned} \pi(x,y) = \frac{1}{4}, \qquad (x,y) \in
                                                    \Sigma_A \times
                            \Sigma_B, \qquad \text{and} \qquad (a, b) \in \Gamma_A \times
                            \Gamma_B,
                    \end{aligned}
            \end{equation}
        \]

        where

        \[
            \begin{equation}
                    \Sigma_A = \{0, 1\}, \quad \Sigma_B = \{0, 1\}, \quad \Gamma_A =
                    \{0,1\}, \quad \text{and} \quad \Gamma_B = \{0, 1\}.
            \end{equation}
        \]

        Alice and Bob win the CHSH game if and only if the following equation is
        satisfied

        \[
            \begin{equation}
            a \oplus b = x \land y.
            \end{equation}
        \]

        Recall that \(\oplus\) refers to the XOR operation.

        The optimal quantum value of CHSH is \(\cos(\pi/8)^2 \approx 0.8536\)
        where the optimal classical value is \(3/4\).

        In order to specify the CHSH game, we can define the probability matrix and
        predicate matrix for the CHSH game as `numpy` arrays as follows.

        ```python exec="1" source="above" result="text" session="chsh"
        import numpy as np

        prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        pred_mat = np.array([[0, 0], [0, 1]])
        ```

        In `toqito`, we can calculate both the quantum and classical value of the
        CHSH game as follows.

        ```python exec="1" source="above" result="text" session="chsh"
        import numpy as np
        from toqito.nonlocal_games.xor_game import XORGame

        chsh = XORGame(prob_mat, pred_mat)

        print(f"Quantum value of the CHSH game is {np.around(chsh.quantum_value(), decimals=2)}")
        print(f"Classical value of the CHSH game is {chsh.classical_value()}")
        ```

        The odd cycle game

        The odd cycle game is another XOR game [@cleve2010consequences]. For this game, we can
        specify the probability and predicate matrices as follows.

        ```python exec="1" source="above" result="text" session="odd_cycle"
        import numpy as np
        prob_mat = np.array(
        [
        [0.1, 0.1, 0, 0, 0],
        [0, 0.1, 0.1, 0, 0],
        [0, 0, 0.1, 0.1, 0],
        [0, 0, 0, 0.1, 0.1],
        [0.1, 0, 0, 0, 0.1],
        ]
        )
        pred_mat = np.array(
        [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        ])
        ```

        In `|toqito⟩`, we can calculate both the quantum and classical value of
        the odd cycle game as follows.

        ```python exec="1" source="above" result="text" session="odd_cycle"
        import numpy as np
        from toqito.nonlocal_games.xor_game import XORGame

        odd_cycle = XORGame(prob_mat, pred_mat)

        print(f"Quantum value of the odd cycle game is {np.around(odd_cycle.quantum_value(), decimals=2)}")
        print(f"Classical value of the odd cycle game is {np.around(odd_cycle.classical_value(), decimals=1)}")
        ```


        We can also calculate the nonsignaling value of the odd cycle game.

        ```python exec="1" source="above" result="text" session="odd_cycle"
        print(f"Nonsignaling value of the odd cycle game is {np.around(odd_cycle.nonsignaling_value(), decimals=1)}")
        ```

    """

    def __init__(
        self,
        prob_mat: np.ndarray,
        pred_mat: np.ndarray,
        reps: int = 1,
        tol: float | None = None,
    ) -> None:
        """Construct XOR game object.

        Raises:
            ValueError: If dimension of probability matrix is not equal to
                                                dimension of predicate matrix.

        Args:
            prob_mat: A matrix whose (q_0, q_1)-entry gives the probability that
                                 the referee will give Alice the value `q_0` and Bob the
                                 value `q_1`.
            pred_mat: A binary matrix whose (q_0, q_1)-entry indicates the
                                 winning choice (either 0 or 1) when Alice and Bob receive
                                 values `q_0` and `q_1` from the referee.
            reps: Number of parallel repetitions to perform. Default is 1.
            tol: The error tolerance for the value.

        """
        self.prob_mat = np.asarray(prob_mat)
        self.pred_mat = np.asarray(pred_mat)
        self.reps = reps

        q_0, q_1 = self.prob_mat.shape
        self.tol = np.finfo(float).eps * q_0**2 * q_1**2 if tol is None else tol

        # Perform some basic error checking to ensure the probability and
        # predicate matrices are well-defined.
        if (q_0, q_1) != self.pred_mat.shape:
            raise ValueError("Invalid: The matrices `prob_mat` and `pred_mat` must be matrices of the same size.")
        if -self.prob_mat.min() > self.tol:
            raise ValueError(
                "Invalid: The variable `prob_mat` must be a probability matrix: its entries must be non-negative."
            )
        if abs(self.prob_mat.sum() - 1) > self.tol:
            raise ValueError(
                "Invalid: The variable `prob_mat` must be a probability matrix: its entries must sum to 1."
            )

    def quantum_value(
        self, solver: str | None = None, solver_params: dict | None = None, return_strategy: bool = False
    ) -> float | tuple[float, QuantumStrategy]:
        r"""Compute the quantum value of the XOR game.

        By Tsirelson's theorem the quantum bias of an XOR game is the value of the semidefinite program (Lecture 6
        of [@watrous2011theory])

        \[
        \begin{equation}
                \begin{aligned}
                        \text{maximize:} \quad & \sum_{x \in X, y \in Y} D(x, y) \langle u_x, v_y \rangle \\
                        \text{subject to:} \quad & \lVert u_x \rVert = \lVert v_y \rVert = 1,
                \end{aligned}
        \end{equation}
        \]

        over real unit vectors, which is solved as a semidefinite program in their Gram matrix. Here \(D\) is the
        matrix

        \[
        D(x,y) = \pi(x, y) (-1)^{f(x,y)},
        \]

        where \(\pi(x, y)\) corresponds to `prob_mat[x, y]` and \(f(x,y)\) corresponds to `pred_mat[x, y]`, and the
        quantum value is \((1 + \text{bias}) / 2\).

        The optimal vectors also give an optimal strategy: with anticommuting Hermitian involutions
        \(C_1, \ldots, C_r\) on \(2^{\lceil r/2 \rceil}\) dimensions, Alice measures the observable
        \(\sum_i u_x(i) C_i\), Bob measures \(\sum_i v_y(i) C_i^{\mathsf{T}}\), and they share the maximally
        entangled state, so that \(\langle A_x \otimes B_y \rangle = \langle u_x, v_y \rangle\).

        Args:
            solver: The cvxpy solver to use (default is cvxpy's choice).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.
            return_strategy: If True, also return the optimal strategy as a `QuantumStrategy` whose measurements
                have outcome ``0`` for the \(+1\) eigenspace of the observable and ``1`` for the \(-1\) eigenspace
                (default is False). Only available for a single repetition.

        Raises:
            ValueError: If a strategy is requested for more than one repetition.

        Returns:
            A value between `[0, 1]` representing the quantum value, followed by the optimal strategy when
            `return_strategy` is True.

        """
        if return_strategy and self.reps != 1:
            raise ValueError("A strategy can only be returned for a single repetition of the game.")
        alice_in, bob_in = self.prob_mat.shape
        d_mat = self.prob_mat * (-1.0) ** self.pred_mat

        # gram[i, j] = <w_i, w_j> for the stacked vectors (u_0, ..., u_{m-1}, v_0, ..., v_{n-1}).
        num_vectors = alice_in + bob_in
        gram = cvxpy.Variable((num_vectors, num_vectors), symmetric=True)
        weights = np.zeros((num_vectors, num_vectors))
        weights[:alice_in, alice_in:] = d_mat / 2
        weights += weights.T
        objective = cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(weights, gram)))
        bias = solve_sdp(cvxpy.Problem(objective, [gram >> 0, cvxpy.diag(gram) == 1]), solver, solver_params)

        # It holds from (https://arxiv.org/abs/quant-ph/0608146) that the quantum value of any XOR game obeys strong
        # parallel repetition, so the value of `reps` parallel copies is the single-copy value to that power.
        value = ((1 + bias) / 2) ** self.reps
        if not return_strategy:
            return value

        # Read the vectors off the Gram matrix, keeping only the directions that carry weight.
        eigvals, eigvecs = np.linalg.eigh(gram.value)
        keep = eigvals > 1e-9 * eigvals.max()
        vectors = eigvecs[:, keep] * np.sqrt(eigvals[keep])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        generators = _clifford_generators(vectors.shape[1])
        dim = generators.shape[1]

        alice_obs = np.einsum("xi,ijk->xjk", vectors[:alice_in], generators)
        bob_obs = np.einsum("yi,ijk->yjk", vectors[alice_in:], generators.transpose(0, 2, 1))
        signs = np.array([1.0, -1.0])
        alice_povms = (np.eye(dim)[None, None] + signs[None, :, None, None] * alice_obs[:, None]) / 2
        bob_povms = (np.eye(dim)[None, None] + signs[None, :, None, None] * bob_obs[:, None]) / 2
        psi = np.eye(dim).reshape(-1) / np.sqrt(dim)
        strategy = QuantumStrategy(
            state=np.outer(psi, psi.conj()),
            alice_povms=alice_povms.transpose(2, 3, 0, 1),
            bob_povms=bob_povms.transpose(2, 3, 0, 1),
        )
        return value, strategy

    def quantum_strategy_value(self, strategy: QuantumStrategy) -> float:
        """Compute the winning probability of a quantum strategy, as for the equivalent `NonlocalGame`.

        Args:
            strategy: A shared state on Alice's and Bob's systems and their measurements.

        Returns:
            The probability of winning the game with the strategy.

        """
        return self.to_nonlocal_game().quantum_strategy_value(strategy)

    def classical_value(self) -> float:
        r"""Compute the classical value of the XOR game.

        Returns:
            A value between `[0, 1]` representing the classical value.

        """
        return self.to_nonlocal_game().classical_value()

    def nonsignaling_value(self) -> float:
        """Compute the nonsignaling value of an XOR game.

        Here, the existing function in the `NonlocalGame` class is called.

        Returns:
            A value between `[0, 1]` representing the nonsignaling value.

        """
        return self.to_nonlocal_game().nonsignaling_value()

    def to_nonlocal_game(self) -> NonlocalGame:
        """Given an XOR game, compute a predicate matrix representing the more generic `NonlocalGame` equivalent.

        The equivalent game is built once and reused by the methods that delegate to it.

        Returns:
            A `NonlocalGame` object equivalent to the XOR game.

        """
        return self._nonlocal_game

    @cached_property
    def _nonlocal_game(self) -> NonlocalGame:
        q_0, q_1 = self.prob_mat.shape
        # Answers (a, b) win on questions (x, y) exactly when a XOR b equals the predicate bit.
        nlg_pred_mat = np.zeros((2, 2, q_0, q_1))
        for a in range(2):
            for b in range(2):
                nlg_pred_mat[a, b] = self.pred_mat == (a ^ b)
        return NonlocalGame(self.prob_mat, nlg_pred_mat, reps=self.reps)
