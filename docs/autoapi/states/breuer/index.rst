states.breuer
=============

.. py:module:: states.breuer

.. autoapi-nested-parse::

   Breuer states represent the Breuer bound entangled states.

   These states are based on the Breuer-Hall criterion.



Functions
---------

.. autoapisummary::

   states.breuer.breuer


Module Contents
---------------

.. py:function:: breuer(dim, lam)

   Produce a Breuer state :cite:`Breuer_2006_Optimal`.

   Gives a Breuer bound entangled state for two qudits of local dimension :code:`dim`, with the
   :code:`lam` parameter describing the weight of the singlet component as described in
   :cite:`Breuer_2006_Optimal`.

   This function was adapted from the QETLAB package.

   .. rubric:: Examples

   We can generate a Breuer state of dimension :math:`4` with weight :math:`0.1`. For any weight above :math:`0`, the
   state will be bound entangled, that is, it will satisfy the PPT criterion, but it will be entangled.

   >>> from toqito.states import breuer
   >>> breuer(2, 0.1)
   array([[0.3, 0. , 0. , 0. ],
          [0. , 0.2, 0.1, 0. ],
          [0. , 0.1, 0.2, 0. ],
          [0. , 0. , 0. , 0.3]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: Dimension must be greater than or equal to 1.
   :param dim: Dimension of the Breuer state.
   :param lam: The weight of the singlet component.
   :return: Breuer state of dimension :code:`dim` with weight :code:`lam`.



