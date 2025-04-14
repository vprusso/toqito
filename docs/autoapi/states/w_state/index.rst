states.w_state
==============

.. py:module:: states.w_state

.. autoapi-nested-parse::

   Generalized w-state is an entangled quantum state of `n` qubits.

   This state refers to the quantum superposition in which one of the qubits is in an excited state and others are in the
   ground state.



Functions
---------

.. autoapisummary::

   states.w_state.w_state


Module Contents
---------------

.. py:function:: w_state(num_qubits, coeff = None)

   Produce a W-state :cite:`Dur_2000_ThreeQubits`.

   Returns the W-state described in :cite:`Dur_2000_ThreeQubits`. The W-state on `num_qubits` qubits is defined by:

   .. math::
       |W \rangle = \frac{1}{\sqrt{num\_qubits}}
       \left(|100 \ldots 0 \rangle + |010 \ldots 0 \rangle + \ldots +
       |000 \ldots 1 \rangle \right).

   .. rubric:: Examples

   Using :code:`|toqito⟩`, we can generate the :math:`3`-qubit W-state

   .. math::
       |W_3 \rangle = \frac{1}{\sqrt{3}} \left( |100\rangle + |010 \rangle +
       |001 \rangle \right)

   as follows.

   >>> from toqito.states import w_state
   >>> w_state(3)
   array([[0.    ],
          [0.5774],
          [0.5774],
          [0.    ],
          [0.5774],
          [0.    ],
          [0.    ],
          [0.    ]])

   We may also generate a generalized :math:`W`-state. For instance, here is a :math:`4`-dimensional :math:`W`-state

   .. math::
       \frac{1}{\sqrt{30}} \left( |1000 \rangle + 2|0100 \rangle + 3|0010
       \rangle + 4 |0001 \rangle \right).

   We can generate this state in :code:`|toqito⟩` as

   >>> from toqito.states import w_state
   >>> import numpy as np
   >>> coeffs = np.array([1, 2, 3, 4]) / np.sqrt(30)
   >>> w_state(4, coeffs)
   array([[0.    ],
          [0.7303],
          [0.5477],
          [0.    ],
          [0.3651],
          [0.    ],
          [0.    ],
          [0.    ],
          [0.1826],
          [0.    ],
          [0.    ],
          [0.    ],
          [0.    ],
          [0.    ],
          [0.    ],
          [0.    ]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: The number of qubits must be greater than or equal to 1.
   :param num_qubits: An integer representing the number of qubits.
   :param coeff: default is `[1, 1, ..., 1]/sqrt(num_qubits)`: a
                 1-by-`num_qubts` vector of coefficients.



