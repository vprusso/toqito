"""Checks if a set of quantum states are distinguishable."""

import numpy as np

from toqito.state_opt.state_distinguishability import state_distinguishability

# The full import path was specified here because the doctest workflow was failing when the above function could not be
# imported https://github.com/vprusso/toqito/issues/473


def is_distinguishable(states: list[np.ndarray], probs: list[float] = None) -> bool:
    r"""Check whether a collection of vectors are (perfectly) distinguishable or not.

    The ability to determine whether a set of quantum states are distinguishable can be obtained via the state
    distinguishability SDP as defined in `state_distinguishability`

    Examples
    ========

    The set of Bell states are an example of distinguishable states. Recall that the Bell states are defined as:

    .. math::
        u_1 = \frac{1}{\sqrt{2}} \left(|00\rangle + |11\rangle\right), &\quad
        u_2 = \frac{1}{\sqrt{2}} \left(|00\rangle - |11\rangle\right), \\
        u_3 = \frac{1}{\sqrt{2}} \left(|01\rangle + |10\rangle\right), &\quad
        u_4 = \frac{1}{\sqrt{2}} \left(|01\rangle - |10\rangle\right).

    It can be checked in :code`toqito` that the Bell states are distinguishable:

    .. jupyter-execute::

        from toqito.states import bell
        from toqito.state_props import is_distinguishable
        bell_states = [bell(0), bell(1), bell(2), bell(3)]
        is_distinguishable(bell_states)

    References
    ==========
    .. footbibliography::



    :param states: A set of vectors consisting of quantum states to determine the distinguishability of.
    :param probs: Respective list of probabilities each state is selected. If no
                probabilities are provided, a uniform probability distribution is assumed.
    :return: :code:`True` if the vectors are distinguishable; :code:`False` otherwise.

    """
    # The dual problem is less computationally intensive to compute in comparison to primal.
    opt_val, _ = state_distinguishability(vectors=states, probs=probs, primal_dual="dual")
    return np.isclose(opt_val, 1)
