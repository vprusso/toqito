state_props.entanglement_of_formation
=====================================

.. py:module:: state_props.entanglement_of_formation

.. autoapi-nested-parse::

   Computes the entanglement of formation of a bipartite quantum state.



Functions
---------

.. autoapisummary::

   state_props.entanglement_of_formation.entanglement_of_formation


Module Contents
---------------

.. py:function:: entanglement_of_formation(rho, dim = None)

   Compute entanglement-of-formation of a bipartite quantum state :cite:`Quantiki_EOF`.

   Entanglement-of-formation is the entropy of formation of the bipartite
   quantum state :code:`rho`. Note that this function currently only supports
   :code:`rho` being a pure state or a 2-qubit state: it is not known how to
   compute the entanglement-of-formation of higher-dimensional mixed states.

   This function was adapted from QETLAB.

   .. rubric:: Examples

   Compute the entanglement-of-formation of a Bell state.

   Let :math:`u = \frac{1}{\sqrt{2}} \left(|00\rangle + |11\rangle \right)`
   and let

   .. math::
       \rho = uu^* = \frac{1}{2}\begin{pmatrix}
                                   1 & 0 & 0 & 1 \\
                                   0 & 0 & 0 & 0 \\
                                   0 & 0 & 0 & 0 \\
                                   1 & 0 & 0 & 1
                                \end{pmatrix}.

   The entanglement-of-formation of :math:`\rho` is equal to 1.

   >>> import numpy as np
   >>> from toqito.state_props import entanglement_of_formation
   >>> from toqito.states import bell
   >>>
   >>> u_vec = bell(0)
   >>> rho = u_vec @ u_vec.conj().T
   >>> np.around(entanglement_of_formation(rho), decimals=3)
   np.float64(1.0)

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If matrices have improper dimension.
   :param rho: A matrix or vector.
   :param dim: The default has both subsystems of equal dimension.
   :return: A value between 0 and 1 that corresponds to the
            entanglement-of-formation of :code:`rho`.



