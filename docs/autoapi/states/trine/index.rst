states.trine
============

.. py:module:: states.trine

.. autoapi-nested-parse::

   Trine states are states of linear polarization separated by 60°.



Functions
---------

.. autoapisummary::

   states.trine.trine


Module Contents
---------------

.. py:function:: trine()

   Produce the set of trine states (Slide 6 of :cite:`Yard_2017_Lecture11`).

   The trine states are formally defined as:

   .. math::
       u_1 = |0\rangle, \quad
       u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
       u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

   .. rubric:: Examples

   Generating the trine states can be done by simply invoking the function:

   >>> from toqito.states import trine
   >>>
   >>> trine()
   [array([[1],
          [0]]), array([[-0.5      ],
          [-0.8660254]]), array([[-0.5      ],
          [ 0.8660254]])]

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :return: Vector of trine states.


