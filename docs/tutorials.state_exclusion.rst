Quantum state exclusion
=======================

In this tutorial, we are going to cover the problem of *quantum state
exclusion*. We are going to briefly describe the problem setting and then
describe how one may use :code:`toqito` to calculate the optimal probability
with which this problem can be solved for a number of different scenarios.

Quantum state exclusion is very closely related to the problem of quantum state
distinguishability. It may be useful to consult the following tutorial that
covers quantum state distinguishability:

* `Quantum State Distinguishability <https://toqito.readthedocs.io/en/latest/tutorials.state_distinguishability.html>`_

Further information beyond the scope of this tutorial can be found in the text
:cite:`Pusey_2012_On` as well as the course :cite:`Bandyopadhyay_2014_Conclusive`.


The state exclusion problem
---------------------------

The quantum state exclusion problem is phrased as follows.

1. Alice possesses an ensemble of :math:`n` quantum states:

    .. math::
        \begin{equation}
            \eta = \left( (p_0, \rho_0), \ldots, (p_n, \rho_n)  \right),
        \end{equation}

where :math:`p_i` is the probability with which state :math:`\rho_i` is
selected from the ensemble. Alice picks :math:`\rho_i` with probability
:math:`p_i` from her ensemble and sends :math:`\rho_i` to Bob.

2. Bob receives :math:`\rho_i`. Both Alice and Bob are aware of how the
   ensemble is defined but he does *not* know what index :math:`i`
   corresponding to the state :math:`\rho_i` he receives from Alice is.

3. Bob wants to guess which of the states from the ensemble he was *not* given.
   In order to do so, he may measure :math:`\rho_i` to guess the index
   :math:`i` for which the state in the ensemble corresponds.

This setting is depicted in the following figure.

.. figure:: figures/quantum_state_distinguish.svg
   :alt: quantum state exclusion
   :align: center

   The quantum state exclusion setting.

.. note::
    The primary difference between the quantum state distinguishability
    scenario and the quantum state exclusion scenario is that in the former,
    Bob want to guess which state he was given, and in the latter, Bob wants to
    guess which state he was *not* given.

Optimal probability of conclusively excluding a quantum state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Optimal probability of unambiguously excluding a quantum state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

References
------------------------------

.. bibliography:: 
    :filter: docname in docnames
