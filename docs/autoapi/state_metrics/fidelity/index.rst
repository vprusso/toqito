state_metrics.fidelity
======================

.. py:module:: state_metrics.fidelity

.. autoapi-nested-parse::

   Fidelity is a metric that qualifies how close two quantum states are.



Functions
---------

.. autoapisummary::

   state_metrics.fidelity.fidelity


Module Contents
---------------

.. py:function:: fidelity(rho, sigma)

   Compute the fidelity of two density matrices :cite:`WikiFidQuant`.

   Calculate the fidelity between the two density matrices :code:`rho` and :code:`sigma`, defined by:

   .. math::
       ||\sqrt(\rho) \sqrt(\sigma)||_1,

   where :math:`|| \cdot ||_1` denotes the trace norm. The return is a value between :math:`0` and :math:`1`, with
   :math:`0` corresponding to matrices :code:`rho` and :code:`sigma` with orthogonal support, and :math:`1`
   corresponding to the case :code:`rho = sigma`.

   .. rubric:: Examples

   Consider the following Bell state

   .. math::
       u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

   The corresponding density matrix of :math:`u` may be calculated by:

   .. math::
       \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                        1 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 \\
                        1 & 0 & 0 & 1
                      \end{pmatrix} \in \text{D}(\mathcal{X}).

   In the event where we calculate the fidelity between states that are identical, we should obtain the value of
   :math:`1`. This can be observed in :code:`|toqito⟩` as follows.

   >>> from toqito.state_metrics import fidelity
   >>> import numpy as np
   >>> rho = 1 / 2 * np.array(
   ...     [[1, 0, 0, 1],
   ...      [0, 0, 0, 0],
   ...      [0, 0, 0, 0],
   ...      [1, 0, 0, 1]]
   ... )
   >>> sigma = rho
   >>> fidelity(rho, sigma)
   np.float64(1.0000000000000002)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrices are not density operators.
   :param rho: Density operator.
   :param sigma: Density operator.
   :return: The fidelity between :code:`rho` and :code:`sigma`.



