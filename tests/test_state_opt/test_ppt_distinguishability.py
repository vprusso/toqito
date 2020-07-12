"""Test ppt_distinguishability."""
import numpy as np

from toqito.perms import swap_operator
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


def test_ppt_distinguishability_werner_hiding_pairs():
    r"""
    One quantum data hiding scheme involves the Werner hiding pair.

    A Werner hiding pair is defined by

    .. math::
    \begin{equation}
        \sigma_0^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} + W_n}{n(n+1)}
        \quad \text{and} \quad
        \sigma_1^{(n)} = \frac{\mathbb{I} \otimes \mathbb{I} - W_n}{n(n-1)}
    \end{equation}

    The optimal probability to distinguish the Werner hiding pair is known
    to be upper bounded by the following equation

    .. math::
    \begin{equation}
        \frac{1}{2} + \frac{1}{n+1}
    \end{equation}
    """
    dim = 2
    sigma_0 = (np.kron(np.identity(dim), np.identity(dim)) + swap_operator(dim)) / (
        dim * (dim + 1)
    )
    sigma_1 = (np.kron(np.identity(dim), np.identity(dim)) - swap_operator(dim)) / (
        dim * (dim - 1)
    )

    states = [sigma_0, sigma_1]

    expected_val = 1 / 2 + 1 / (dim + 1)
    res = ppt_distinguishability(states)
    np.testing.assert_equal(np.isclose(res, expected_val), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
