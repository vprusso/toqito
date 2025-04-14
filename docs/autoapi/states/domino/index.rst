states.domino
=============

.. py:module:: states.domino

.. autoapi-nested-parse::

   Produce a domino state.



Functions
---------

.. autoapisummary::

   states.domino.domino


Module Contents
---------------

.. py:function:: domino(idx)

   Produce a domino state :cite:`Bennett_1999_QuantumNonlocality, Bennett_1999_UPB`.

   The orthonormal product basis of domino states is given as

   .. math::
       \begin{equation}
           \begin{aligned}
           |\phi_0\rangle = |1\rangle |1 \rangle,
           \qquad
           |\phi_1\rangle = |0 \rangle \left(\frac{|0 \rangle + |1 \rangle}{\sqrt{2}} \right),
           & \qquad
           |\phi_2\rangle = |0\rangle \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right), \\
           |\phi_3\rangle = |2\rangle \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right), \qquad
           |\phi_4\rangle = |2\rangle \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right), & \qquad
           |\phi_5\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right) |0\rangle, \\
           |\phi_6\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right) |0\rangle, \qquad
           |\phi_7\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right) |2\rangle, & \qquad
           |\phi_8\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right) |2\rangle.
           \end{aligned}
       \end{equation}

   Returns one of the following nine domino states depending on the value of :code:`idx`.

   .. rubric:: Examples

   When :code:`idx = 0`, this produces the following Domino state

   .. math::
       |\phi_0 \rangle = |11 \rangle |11 \rangle.

   Using :code:`|toqito⟩`, we can see that this yields the proper state.

   >>> from toqito.states import domino
   >>> domino(0)
   array([[0],
          [0],
          [0],
          [0],
          [1],
          [0],
          [0],
          [0],
          [0]])

   When :code:`idx = 3`, this produces the following Domino state

   .. math::
       |\phi_3\rangle = |2\rangle \left(\frac{|0\rangle + |1\rangle}
       {\sqrt{2}}\right)

   Using :code:`|toqito⟩`, we can see that this yields the proper state.

   >>> from toqito.states import domino
   >>> domino(3)
   array([[0.        ],
          [0.        ],
          [0.        ],
          [0.        ],
          [0.        ],
          [0.        ],
          [0.        ],
          [0.70710678],
          [0.70710678]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: Invalid value for :code:`idx`.
   :param idx: A parameter in [0, 1, 2, 3, 4, 5, 6, 7, 8]
   :return: Domino state of index :code:`idx`.



