Superdense Coding
==================

In classical communication, sending two bits of information requires transmitting
two physical bits. But with the help of quantum entanglement, we can bend this rule. 

**Superdense coding** proposed bye coding** proposed by Bennet and Wiesner in 1992 
:cite:`Bennett_1992_Communication` lets Alice send two classical bits to
Bob by transmitting just *one qubit*. The catch here is that they must share an
entangled pair of qubits beforehand. We will explain this protocol in detail 
below:

Superdense coding protocol
^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Before any communication begins, a third party prepares two qubits in 
   *Bell state*:

   .. math::

      \begin{equation}
          \begin{aligned}
              \ket{\psi} = \frac{\ket{00} + \ket{11}}{\sqrt{2}}
          \end{aligned}
      \end{equation}

   Alice takes the first qubit, Bob takes the second, and they both separate.
   This entangled pair is responsible for linking qubit *non-locally*, allowing
   Alice's local operations to affect the global state.

   .. code-block:: python

       import numpy as np
       from toqito.states import bell
       from toqito.matrices import pauli, cnot, hadamard
       
       bell_state = bell(0)
       print("Initial Bell state (|Φ⁺⟩):\n", bell_state)

2. Alice holds two classical bits (:math:`a` and :math:`b`) that she wants to send.
   Depending on the values of her classical bits, she applies one of the four *Pauli Gates* 
   to her qubit for encoding:

.. raw:: html

   <div style="text-align: center;">

.. list-table:: 
   :header-rows: 1
   :widths: 20 20 40 60 100
   
   * - :math:`a`
     - :math:`b`
     - *message*
     - *Gate applied*
     - *Final output (Bell state)*
   * - :math:`0`
     - :math:`0`
     - :math:`\ket{00}`
     - :math:`I`
     - :math:`\frac{|00\rangle + |11\rangle}{\sqrt{2}}`
   * - :math:`0`
     - :math:`1`
     - :math:`\ket{01}`
     - :math:`X`
     - :math:`\frac{|10\rangle + |01\rangle}{\sqrt{2}}`
   * - :math:`1`
     - :math:`0`
     - :math:`\ket{10}`
     - :math:`Z`
     - :math:`\frac{|00\rangle - |11\rangle}{\sqrt{2}}`
   * - :math:`1`
     - :math:`1`
     - :math:`\ket{11}`
     - :math:`XZ = iY`
     - :math:`\frac{|10\rangle - |01\rangle}{\sqrt{2}}`

.. raw:: html

   </div>

.. code-block:: python

    pauli_gate_operations = {
        # Identity gate.
        "00": pauli("I"),
        # Pauli-X gate.
        "01": pauli("X"),
        # Pauli-Z gate.
        "10": pauli("Z"),
        # X followed by Z (equivalent to iY).
        "11": 1j * pauli("Y")  
    }

    message_to_encode = "11"

    entangled_state_encoded = np.kron(pauli_gate_operations[message_to_encode], pauli("I")) @ bell_state

3. Bob performs operations to reverse the entanglement sent by Alice and extract the bits. First, 
   he applies a Controlled-NOT or :math:`CX` *(CNOT) Gate* with the qubit received from Alice as the
   *control qubit* and Bob's original qubit as the *target qubit*. After this, Bob moves
   ahead and applies a Hadamard or :math:`H` gate to Alice's qubit.

   .. code-block:: python

       state_after_cnot = cnot() @ entangled_state_encoded
       decoded_state = np.kron(hadamard(1), pauli("I")) @ state_after_cnot
       print("Decoded state:\n", decoded_state)

4. Finally, Bob measures both qubits in the computational basis (:math:`\ket{0}, 
   \ket{1}`). The result is guaranteed to be :math:`11`; the two bits that Alice sent.

   .. code-block:: python

       measurement_probabilities = np.abs(decoded_state.flatten())**2
       print("Measurement probabilities for basis states |00>, |01>, |10>, |11>:")
       print(measurement_probabilities)


Full code:

.. code-block:: python

  >>> import numpy as np
  >>> from toqito.states import bell
  >>> from toqito.matrices import pauli, cnot, hadamard
  >>> np.set_printoptions(precision=8, suppress=True)
  >>>
  >>> bell_state = bell(0)
  >>> print("Initial Bell state (|Φ⁺⟩):\n", bell_state)
  Initial Bell state (|Φ⁺⟩):
   [[0.70710678]
   [0.        ]
   [0.        ]
   [0.70710678]]

  >>> pauli_gate_operations = {
  ...     "00": pauli("I"),
  ...     "01": pauli("X"),
  ...     "10": pauli("Z"),
  ...     "11": 1j * pauli("Y")
  ... }

  >>> message_to_encode = "10"

  >>> entangled_state_encoded = np.kron(pauli_gate_operations[message_to_encode], pauli("I")) @ bell_state

  >>> state_after_cnot = cnot() @ entangled_state_encoded

  >>> decoded_state = np.kron(hadamard(1), pauli("I")) @ state_after_cnot

  >>> measurement_probabilities = np.abs(decoded_state.flatten())**2
  >>> print("Measurement probabilities for basis states |00>, |01>, |10>, |11>:\n", measurement_probabilities)
  Measurement probabilities for basis states |00>, |01>, |10>, |11>:
   [0. 0. 1. 0.]
  


References
------------------------------

.. bibliography:: 
    :filter: docname in docnames