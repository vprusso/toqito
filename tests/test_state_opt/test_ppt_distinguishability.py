"""Test ppt_distinguishability."""
import numpy as np

from toqito.state_opt import ppt_distinguishability
from toqito.states import bell


def test_ppt_distinguishability_yyd_density_matrices():
    """
    PPT distinguishing the YYD states from [1] should yield `7/8 ~ 0.875`

    Feeding the input to the function as density matrices.

    References:
    [1]: Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
    "Four locally indistinguishable ququad-ququad orthogonal
    maximally entangled states."
    Physical review letters 109.2 (2012): 020506.
    https://arxiv.org/abs/1107.3224
    """
    psi_0 = bell(0)
    psi_1 = bell(2)
    psi_2 = bell(3)
    psi_3 = bell(1)

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    rho_1 = x_1 * x_1.conj().T
    rho_2 = x_2 * x_2.conj().T
    rho_3 = x_3 * x_3.conj().T
    rho_4 = x_4 * x_4.conj().T

    states = [rho_1, rho_2, rho_3, rho_4]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    res = ppt_distinguishability(states, probs)
    np.testing.assert_equal(np.isclose(res, 7 / 8), True)


def test_ppt_distinguishability_yyd_vectors():
    """
    PPT distinguishing the YYD states from [1] should yield `7/8 ~ 0.875`

    Feeding the input to the function as state vectors.

    References:
    [1]: Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
    "Four locally indistinguishable ququad-ququad orthogonal
    maximally entangled states."
    Physical review letters 109.2 (2012): 020506.
    https://arxiv.org/abs/1107.3224
    """
    psi_0 = bell(0)
    psi_1 = bell(2)
    psi_2 = bell(3)
    psi_3 = bell(1)

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    states = [x_1, x_2, x_3, x_4]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    res = ppt_distinguishability(states, probs)
    np.testing.assert_equal(np.isclose(res, 7 / 8), True)


def test_ppt_distinguishability_yyd_states_no_probs():
    """
    PPT distinguishing the YYD states from [1] should yield 7/8 ~ 0.875

    If no probability vector is explicitly given, assume uniform
    probabilities are given.

    References:
    [1]: Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
    "Four locally indistinguishable ququad-ququad orthogonal
    maximally entangled states."
    Physical review letters 109.2 (2012): 020506.
    https://arxiv.org/abs/1107.3224
    """
    psi_0 = bell(0)
    psi_1 = bell(2)
    psi_2 = bell(3)
    psi_3 = bell(1)

    x_1 = np.kron(psi_0, psi_0)
    x_2 = np.kron(psi_1, psi_3)
    x_3 = np.kron(psi_2, psi_3)
    x_4 = np.kron(psi_3, psi_3)

    rho_1 = x_1 * x_1.conj().T
    rho_2 = x_2 * x_2.conj().T
    rho_3 = x_3 * x_3.conj().T
    rho_4 = x_4 * x_4.conj().T

    states = [rho_1, rho_2, rho_3, rho_4]

    res = ppt_distinguishability(states)
    np.testing.assert_equal(np.isclose(res, 7 / 8), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
