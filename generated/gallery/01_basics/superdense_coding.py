"""# Superdense Coding"""

# %%
# In classical communication, sending two bits of information requires transmitting
# two physical bits. But with the help of quantum entanglement, we can bend this rule.
#
# **Superdense coding** proposed by Bennet and Wiesner in 1992
# [@Bennett_1992_Communication] lets Alice send two classical bits to
# Bob by transmitting just *one qubit*. The catch here is that they must share an
# entangled pair of qubits beforehand. We will explain this protocol in detail
# below:
#
# ## Superdense coding protocol
#
# **1.** Before any communication begins, a third party prepares two qubits in
#    *Bell state*:
#
# $$
# \ket{\psi} = \frac{\ket{00} + \ket{11}}{\sqrt{2}}
# $$
#
# Alice takes the first qubit, Bob takes the second, and they both separate.
# This entangled pair is responsible for linking the qubits *non-locally*,
# allowing Alice's local operations to affect the global state.
#
import numpy as np

from toqito.matrices import cnot, hadamard, pauli
from toqito.states import bell

np.set_printoptions(precision=8, suppress=True)

bell_state = bell(0)
print(f"Initial Bell state (|Φ⁺⟩): \n {bell_state}")

# %%
# **2.** Alice holds two classical bits ($a$ and $b$) that she wants to
#    send. For the tutorial, she is choosing to send $11$.
#    Depending on the values of her classical bits, she applies one of the four
#    *Pauli Gates* to her qubit for encoding:
#
# | $a$ | $b$ | *message* | *Gate applied* | *Final output (Bell state)* |
# |:---:|:---:|:---------:|:--------------:|:---------------------------:|
# | $0$ | $0$ | $\ket{00}$ | $I$ | $\frac{\ket{00} + \ket{11}}{\sqrt{2}}$ |
# | $0$ | $1$ | $\ket{01}$ | $X$ | $\frac{\ket{10} + \ket{01}}{\sqrt{2}}$ |
# | $1$ | $0$ | $\ket{10}$ | $Z$ | $\frac{\ket{00} - \ket{11}}{\sqrt{2}}$ |
# | $1$ | $1$ | $\ket{11}$ | $XZ = iY$ | $\frac{\ket{10} - \ket{01}}{\sqrt{2}}$ |
#
pauli_gate_operations = {
    # Identity gate.
    "00": pauli("I"),
    # Pauli-X gate.
    "01": pauli("X"),
    # Pauli-Z gate.
    "10": pauli("Z"),
    # X followed by Z (equivalent to iY).
    "11": 1j * pauli("Y"),
}

message_to_encode = "11"

# Alice sends her encoded entangled state after this step.
entangled_state_encoded = np.kron(pauli_gate_operations[message_to_encode], pauli("I")) @ bell_state
print(f"Entangled state is: {entangled_state_encoded}")


# %%
# **3.** Bob performs operations to reverse the entanglement on encoded state sent
#    by Alice and extract the bits. First, he applies a Controlled-NOT or
#    $CX$ *(CNOT) Gate* with the qubit received from Alice as the *control
#    qubit* and Bob's original qubit as the *target qubit*. After this, Bob
#    moves ahead and applies a Hadamard or $H$ gate to Alice's qubit.
#
state_after_cnot = cnot() @ entangled_state_encoded
decoded_state = np.kron(hadamard(1), pauli("I")) @ state_after_cnot
print(f"Decoded state:\n {decoded_state}")

# %%
# **4.** Finally, Bob measures both qubits in the computational basis
#    ($\ket{0}, \ket{1}$). The result is guaranteed to be $11$; the
#    two bits that Alice sent.
#
measurement_probabilities = np.abs(decoded_state.flatten()) ** 2
print(f"Measurement probabilities for basis states |00>, |01>, |10>, |11>: \n {measurement_probabilities}")
# mkdocs_gallery_thumbnail_path = 'figures/logo.png'
# %%
