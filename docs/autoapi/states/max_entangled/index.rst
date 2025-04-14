states.max_entangled
====================

.. py:module:: states.max_entangled

.. autoapi-nested-parse::

   Maximally entangled states are states where the qubits are completely dependent on each other.

   In these states, when a measurement is taken on one of the qubits, the state of the other qubits is automatically known.



Functions
---------

.. autoapisummary::

   states.max_entangled.max_entangled


Module Contents
---------------

.. py:function:: max_entangled(dim, is_sparse = False, is_normalized = True)

   Produce a maximally entangled bipartite pure state :cite:`WikiMaxEnt`.

   Produces a maximally entangled pure state as above that is sparse if :code:`is_sparse = True` and is full if
   :code:`is_sparse = False`. The pure state is normalized to have Euclidean norm 1 if :code:`is_normalized = True`,
   and it is unnormalized (i.e. each entry in the vector is 0 or 1 and the Euclidean norm of the vector is
   :code:`sqrt(dim)` if :code:`is_normalized = False`.

   .. rubric:: Examples

   We can generate the canonical :math:`2`-dimensional maximally entangled state

   .. math::
       u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right)

   using :code:`|toqito⟩` as follows.

   >>> from toqito.states import max_entangled
   >>> max_entangled(2)
   array([[0.70710678],
          [0.        ],
          [0.        ],
          [0.70710678]])

   By default, the state returned in normalized, however we can generate the unnormalized state

   .. math::
       v = |00\rangle + |11 \rangle

   using :code:`|toqito⟩` as follows.

   >>> from toqito.states import max_entangled
   >>> max_entangled(2, False, False)
   array([[1.],
          [0.],
          [0.],
          [1.]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :param dim: Dimension of the entangled state.
   :param is_sparse: `True` if vector is sparse and `False` otherwise.
   :param is_normalized: `True` if vector is normalized and `False` otherwise.
   :return: The maximally entangled state of dimension :code:`dim`.



