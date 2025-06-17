"""Check if set of states are antidistinguishable."""

import numpy as np

from toqito.state_opt.state_exclusion import state_exclusion

# The full import path was specified here because the doctest workflow was failing when the above function could not be
# imported https://github.com/vprusso/toqito/issues/473


def is_antidistinguishable(states: list[np.ndarray]) -> bool:
    r"""Check whether a collection of vectors are antidistinguishable or not.

    For more information, see :footcite:`Heinosaari_2018_Antidistinguishability`.

    The ability to determine whether a set of quantum states are antidistinguishable can be obtained via the state
    exclusion SDP :footcite:`Bandyopadhyay_2014_Conclusive` such that we ignore the associated probabilities with which
    the states are chosen from the set of vectors.

    Examples
    ========

    The set of Bell states are an example of antidistinguishable states. Recall that the Bell states are defined as:

    .. math::
        u_1 = \frac{1}{\sqrt{2}} \left(|00\rangle + |11\rangle\right), &\quad
        u_2 = \frac{1}{\sqrt{2}} \left(|00\rangle - |11\rangle\right), \\
        u_3 = \frac{1}{\sqrt{2}} \left(|01\rangle + |10\rangle\right), &\quad
        u_4 = \frac{1}{\sqrt{2}} \left(|01\rangle - |10\rangle\right).

    It can be checked in :code`toqito` that the Bell states are antidistinguishable:

    .. jupyter-execute::

        from toqito.states import bell
        from toqito.state_props import is_antidistinguishable
        bell_states = [bell(0), bell(1), bell(2), bell(3)]
        is_antidistinguishable(bell_states)

    Consider the following measurement operators

    .. math::
        M_i = \frac{1}{3}\left(\mathbb{I}_{\mathcal{X} - u_i u_i^*\right)

    for all :math:`1 \leq i \leq 4`. It can be verified that these constitute a valid set of POVMs, that is
    :math:`\sum_{i=1}^4 M_i = \mathbb{I}_{\mathcal{X}}` and :math:`M_i \in \text{Pos}(\mathcal{X})` for all :math:`1
    \leq i \leq 4`. It may also be verified that

    .. math::
        \sum_{i=1}^4 \langle M_i, u_i u_i^* \rangle = 0,

    and hence, the Bell states are antidistinguishable.

    References
    ==========
    .. footbibliography::



    :param states: A set of vectors consisting of quantum states to determine the antidistinguishability of.
    :return: :code:`True` if the vectors are antidistinguishable; :code:`False` otherwise.

    """
    probs = [1] * len(states)

    # The dual problem is less computationally intensive to compute in comparison to primal.
    opt_val, _ = state_exclusion(vectors=states, probs=probs, primal_dual="dual")
    return np.isclose(opt_val, 0)
