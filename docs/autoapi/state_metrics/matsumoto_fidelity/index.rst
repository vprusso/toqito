state_metrics.matsumoto_fidelity
================================

.. py:module:: state_metrics.matsumoto_fidelity

.. autoapi-nested-parse::

   Matsumoto fidelity is the maximum classical fidelity associated with a classical-to-quantum preparation procedure.



Functions
---------

.. autoapisummary::

   state_metrics.matsumoto_fidelity.matsumoto_fidelity


Module Contents
---------------

.. py:function:: matsumoto_fidelity(rho, sigma)

   Compute the Matsumoto fidelity of two density matrices :cite:`Matsumoto_2010_Reverse`.

   Calculate the Matsumoto fidelity between the two density matrices :code:`rho` and :code:`sigma`, defined by:

   .. math::
       \mathrm{tr}(\rho\#\sigma),

   where :math:`\#` denotes the matrix geometric mean, which for invertible states is

   .. math::
       \rho\#\sigma = \rho^{1/2}\sqrt{\rho^{-1/2}\sigma\rho^{-1/2}}\rho^{1/2}.

   For singular states it is defined by the limit

   .. math::
       \rho\#\sigma = \lim_{\epsilon\to0}(\rho+\epsilon\mathbb{I})\#(+\epsilon\mathbb{I}).

   The return is a value between :math:`0` and :math:`1`, with :math:`0` corresponding to matrices :code:`rho` and
   :code:`sigma` with orthogonal support, and :math:`1` corresponding to the case :code:`rho = sigma`. The Matsumoto
   fidelity is a lower bound for the fidelity.

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

   In the event where we calculate the Matsumoto fidelity between states that are identical, we should obtain the value
   of :math:`1`. This can be observed in :code:`|toqito⟩` as follows.

   >>> from toqito.state_metrics import matsumoto_fidelity
   >>> import numpy as np
   >>> rho = 1 / 2 * np.array(
   ...     [[1, 0, 0, 1],
   ...      [0, 0, 0, 0],
   ...      [0, 0, 0, 0],
   ...      [1, 0, 0, 1]]
   ... )
   >>> sigma = rho
   >>> np.around(matsumoto_fidelity(rho, sigma), decimals=2)
   np.float64(1.0)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrices are not of equal dimension.
   :param rho: Density operator.
   :param sigma: Density operator.
   :return: The Matsumoto fidelity between :code:`rho` and :code:`sigma`.



