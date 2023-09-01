"""Test ppt_distinguishability."""
import numpy as np

from toqito.perms import swap_operator
from toqito.state_opt import ppt_distinguishability
from toqito.states import basis, bell


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

    # Min-error tests:
    primal_res = ppt_distinguishability(states, probs=probs, dist_method="min-error", strategy=True)
    dual_res = ppt_distinguishability(states, probs=probs, dist_method="min-error", strategy=False)

    np.testing.assert_equal(np.isclose(primal_res, 7 / 8, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, 7 / 8, atol=0.001), True)

    primal_res = ppt_distinguishability(
        states, probs=probs, dist_method="unambiguous", strategy=True
    )
    dual_res = ppt_distinguishability(
        states, probs=probs, dist_method="unambiguous", strategy=False
    )

    np.testing.assert_equal(np.isclose(primal_res, 3 / 4, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, 3 / 4, atol=0.001), True)


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

    primal_res = ppt_distinguishability(states, probs=probs, dist_method="min-error", strategy=True)
    dual_res = ppt_distinguishability(states, probs=probs, dist_method="min-error", strategy=False)

    np.testing.assert_equal(np.isclose(primal_res, 7 / 8, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, 7 / 8, atol=0.001), True)

    primal_res = ppt_distinguishability(
        states, probs=probs, dist_method="unambiguous", strategy=True
    )
    dual_res = ppt_distinguishability(
        states, probs=probs, dist_method="unambiguous", strategy=False
    )

    np.testing.assert_equal(np.isclose(primal_res, 3 / 4, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, 3 / 4, atol=0.001), True)


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

    primal_res = ppt_distinguishability(states, probs=None, dist_method="min-error", strategy=True)
    dual_res = ppt_distinguishability(states, probs=None, dist_method="min-error", strategy=False)

    np.testing.assert_equal(np.isclose(primal_res, 7 / 8, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, 7 / 8, atol=0.001), True)

    primal_res = ppt_distinguishability(
        states, probs=None, dist_method="unambiguous", strategy=True
    )
    dual_res = ppt_distinguishability(states, probs=None, dist_method="unambiguous", strategy=False)

    np.testing.assert_equal(np.isclose(primal_res, 3 / 4, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, 3 / 4, atol=0.001), True)


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

    References:
    [1]: Terhal, Barbara M., David P. DiVincenzo, and Debbie W. Leung.
    "Hiding bits in Bell states."
    Physical review letters 86.25 (2001): 5807.
    https://arxiv.org/abs/quant-ph/0011042

    [2]: Cosentino, Alessandro
    "Quantum state local distinguishability via convex optimization".
    University of Waterloo, Thesis
    https://uwspace.uwaterloo.ca/handle/10012/9572
    """
    dim = 2
    sigma_0 = (np.kron(np.identity(dim), np.identity(dim)) + swap_operator(dim)) / (dim * (dim + 1))
    sigma_1 = (np.kron(np.identity(dim), np.identity(dim)) - swap_operator(dim)) / (dim * (dim - 1))

    states = [sigma_0, sigma_1]

    expected_val = 1 / 2 + 1 / (dim + 1)

    primal_res = ppt_distinguishability(states, probs=None, dist_method="min-error", strategy=True)
    dual_res = ppt_distinguishability(states, probs=None, dist_method="min-error", strategy=False)

    np.testing.assert_equal(np.isclose(primal_res, expected_val, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, expected_val, atol=0.001), True)

    primal_res = ppt_distinguishability(
        states, probs=None, dist_method="unambiguous", strategy=True
    )
    dual_res = ppt_distinguishability(states, probs=None, dist_method="unambiguous", strategy=False)

    np.testing.assert_equal(np.isclose(primal_res, 1 / 3, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, 1 / 3, atol=0.001), True)


def test_ppt_distinguishability_four_bell_states():
    r"""
    PPT distinguishing the four Bell states.

    There exists a closed form formula for the probability with which one
    is able to distinguish one of the four Bell states given with equal
    probability when Alice and Bob have access to a resource state [1].

    The resource state is defined by

    ..math::
        |\tau_{\epsilon} \rangle = \sqrt{\frac{1+\epsilon}{2}} +
        |0\rangle | 0\rangle +
        \sqrt{\frac{1-\epsilon}{2}} |1 \rangle |1 \rangle

    The closed form probability with which Alice and Bob can distinguish via
    PPT measurements is given as follows

    .. math::
        \frac{1}{2} \left(1 + \sqrt{1 - \epsilon^2} \right).

    This formula happens to be equal to LOCC and SEP as well for this case.
    Refer to Theorem 5 in [1] for more details.

    References:
    [1]: Bandyopadhyay, Somshubhro, et al.
    "Limitations on separable measurements by convex optimization."
    IEEE Transactions on Information Theory 61.6 (2015): 3593-3604.
    https://arxiv.org/abs/1408.6981
    """
    rho_1 = bell(0) * bell(0).conj().T
    rho_2 = bell(1) * bell(1).conj().T
    rho_3 = bell(2) * bell(2).conj().T
    rho_4 = bell(3) * bell(3).conj().T

    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    eps = 0.5
    resource_state = np.sqrt((1 + eps) / 2) * e_00 + np.sqrt((1 - eps) / 2) * e_11
    resource_state = resource_state * resource_state.conj().T

    states = [
        np.kron(rho_1, resource_state),
        np.kron(rho_2, resource_state),
        np.kron(rho_3, resource_state),
        np.kron(rho_4, resource_state),
    ]
    probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    exp_res = 1 / 2 * (1 + np.sqrt(1 - eps ** 2))

    primal_res = ppt_distinguishability(states, probs=probs, dist_method="min-error", strategy=True)
    dual_res = ppt_distinguishability(states, probs=probs, dist_method="min-error", strategy=False)

    np.testing.assert_equal(np.isclose(primal_res, exp_res, atol=0.001), True)
    np.testing.assert_equal(np.isclose(dual_res, exp_res, atol=0.001), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
