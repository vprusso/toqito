states.dicke
============

.. py:module:: states.dicke

.. autoapi-nested-parse::

   Dicke states are an equal-weight superposition of all n-qubit states with Hamming Weight k.



Functions
---------

.. autoapisummary::

   states.dicke.dicke


Module Contents
---------------

.. py:function:: dicke(num_qubit, num_excited, return_dm = False)

   Produce a Dicke state with specified excitations.

   The Dicke state is a quantum state with a fixed number of excitations (i.e., `num_excited`)
   distributed across the given number of qubits (i.e., `num_qubit`). It is symmetric and represents
   an equal superposition of all possible states with the specified number of excited qubits.

   .. rubric:: Example

   Consider generating a Dicke state with 3 qubits and 1 excitation:

   >>> from toqito.states import dicke
   >>> dicke(3, 1)
   array([0.        , 0.57735027, 0.57735027, 0.        , 0.57735027,
          0.        , 0.        , 0.        ])

   If we request the density matrix for this state, the return value is:

   >>> from toqito.states import dicke
   >>> dicke(3, 1, return_dm=True)
   array([[0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ],
          [0.        , 0.33333333, 0.33333333, 0.        , 0.33333333,
           0.        , 0.        , 0.        ],
          [0.        , 0.33333333, 0.33333333, 0.        , 0.33333333,
           0.        , 0.        , 0.        ],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ],
          [0.        , 0.33333333, 0.33333333, 0.        , 0.33333333,
           0.        , 0.        , 0.        ],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ],
          [0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ]])

   .. rubric:: References

   .. bibliography::
       :filter: docname in docnames

   :raises ValueError: If the number of excitations exceeds the number of qubits.
   :param num_qubit: The total number of qubits in the system.
   :param num_excited: The number of qubits that are in the excited state.
   :param return_dm: If True, returns the state as a density matrix (default is False).

   :return: The Dicke state vector or density matrix as a NumPy array.


