states.bell
===========

.. py:module:: states.bell

.. autoapi-nested-parse::

   Bell states represent teh simplest examples of quantum entanglement of two qubits.

   Also known as EPR pairs, Bell states comprise of four quantum states in a superposition of 0 and 1.



Functions
---------

.. autoapisummary::

   states.bell.bell


Module Contents
---------------

.. py:function:: bell(idx)

   Produce a Bell state :cite:`WikiBellSt`.

   Returns one of the following four Bell states depending on the value of :code:`idx`:

   .. math::
       \begin{equation}
           \begin{aligned}
               u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right), &
               \qquad &
               u_1 = \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right), \\
               u_2 = \frac{1}{\sqrt{2}} \left( |01 \rangle + |10 \rangle \right), &
               \qquad &
               u_3 = \frac{1}{\sqrt{2}} \left( |01 \rangle - |10 \rangle \right).
           \end{aligned}
       \end{equation}

   .. rubric:: Examples

   When :code:`idx = 0`, this produces the following Bell state:

   .. math::
       u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

   Using :code:`|toqito⟩`, we can see that this yields the proper state.

   >>> from toqito.states import bell
   >>> import numpy as np
   >>> bell(0)
   array([[0.70710678],
          [0.        ],
          [0.        ],
          [0.70710678]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If :code:`idx` is not an integer.
   :param idx: A parameter in [0, 1, 2, 3]
   :return: Bell state with index :code:`idx`.


