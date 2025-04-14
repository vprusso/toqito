state_props.purity
==================

.. py:module:: state_props.purity

.. autoapi-nested-parse::

   Calcultes the purity of a quantum state.



Functions
---------

.. autoapisummary::

   state_props.purity.purity


Module Contents
---------------

.. py:function:: purity(rho)

   Compute the purity of a quantum state :cite:`WikiPurity`.

   The negativity of a subsystem can be defined in terms of a density matrix :math:`\rho`: The
   purity of a quantum state :math:`\rho` is defined as

   .. math::
       \text{Tr}(\rho^2),

   where :math:`\text{Tr}` is the trace function.

   .. rubric:: Examples

   Consider the following scaled state defined as the scaled identity matrix

   .. math::
       \rho = \frac{1}{4} \begin{pmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 \\
                        0 & 0 & 1 & 0 \\
                        0 & 0 & 0 & 1
                      \end{pmatrix} \in \text{D}(\mathcal{X}).

   Calculating the purity of :math:`\rho` yields :math:`\frac{1}{4}`. This can be observed using
   :code:`|toqito⟩` as follows.

   >>> from toqito.state_props import purity
   >>> import numpy as np
   >>> purity(np.identity(4) / 4)
   np.float64(0.25)

   Calculate the purity of the Werner state:

   >>> from toqito.states import werner
   >>> rho = werner(2, 1 / 4)
   >>> np.around(purity(rho), decimals=4)
   np.float64(0.2653)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrix is not density operator.
   :param rho: A density matrix of a pure state vector.
   :return: A value between 0 and 1 that corresponds to the purity of
           :math:`\rho`.



