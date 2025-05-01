# ruff: noqa: D205, D400, D415
"""BB84 Quantum Key Distribution Protocol
======================================

This example demonstrates the BB84 quantum key distribution protocol developed by
Bennett and Brassard in 1984. The protocol allows two parties to establish a
shared secret key with security guaranteed by quantum mechanics.
"""


import numpy as np
from toqito.states import basis
import matplotlib.pyplot as plt

# %%
# Introduction to the BB84 Protocol
# ---------------------------------
#
# The BB84 protocol was the first quantum cryptographic protocol, developed by
# Charles Bennett and Gilles Brassard in 1984. It allows two parties (Alice and Bob)
# to establish a shared secret key using quantum communication, with security
# guaranteed by the laws of quantum mechanics.
#
# The protocol's security is based on two fundamental principles of quantum mechanics:
#
# 1. The no-cloning theorem: It's impossible to create a perfect copy of an unknown quantum state.
# 2. The measurement disturbance principle: Measuring a quantum system generally disturbs it.
#
# These properties allow the detection of any eavesdropper who intercepts the quantum
# communication.
#
# The protocol works as follows:
#
# .. math::
#    \begin{align}
#    &\text{1. Alice generates random bits and random bases (computational }  \text{ or Hadamard } \text{)}\\
#    &\text{2. Alice prepares qubits according to her bits and bases}\\
#    &\text{3. Bob measures each qubit in a randomly chosen basis}\\
#    &\text{4. Alice and Bob publicly compare their bases and keep only matching results}\\
#    &\text{5. They check a subset of their key to detect eavesdropping}
#    \end{align}


# %%
# BB84 Protocol Implementation
# ----------------------------
#
# Let's implement a simulation of the BB84 protocol:

def run_bb84_protocol(num_qubits=100, error_rate=0, eavesdropper=False):
    """
    Simulate the BB84 quantum key distribution protocol.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits to use in the protocol.
    error_rate : float
        Probability of measurement error.
    eavesdropper : bool
        Whether to simulate an eavesdropper.
        
    Returns
    -------
    tuple
        Alice's key, Bob's key, and the error rate between them.
    """
    # Alice's random bits and bases
    alice_bits = np.random.randint(0, 2, num_qubits)
    alice_bases = np.random.randint(0, 2, num_qubits)  # 0: standard, 1: Hadamard
    
    # Bob's random bases
    bob_bases = np.random.randint(0, 2, num_qubits)
    
    # Transmission and measurement
    bob_results = []
    
    for i in range(num_qubits):
        # Alice prepares qubit
        if alice_bits[i] == 0:
            state = basis(2, 0)  # |0⟩
        else:
            state = basis(2, 1)  # |1⟩
        
        # Ensure state is a column vector
        state = np.asarray(state).flatten()
        
        # Alice encodes in chosen basis
        if alice_bases[i] == 1:
            # Hadamard transformation
            h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            state = h_gate @ state
        
        # Eve intercepts
        if eavesdropper:
            eve_basis = np.random.randint(0, 2)
            # Eve measures in her chosen basis
            if eve_basis == 1:
                h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                eve_state = h_gate @ state
            else:
                eve_state = state
                
            # Eve's measurement collapses the state
            eve_probs = np.array([abs(eve_state[0])**2, abs(eve_state[1])**2])
            # Normalize to handle numerical precision issues
            eve_probs = eve_probs / np.sum(eve_probs)
            
            # Eve's measurement outcome
            eve_outcome = np.random.choice([0, 1], p=eve_probs)
            
            if eve_basis == 1:
                h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                state = h_gate @ basis(2, eve_outcome).flatten()
            else:
                state = basis(2, eve_outcome).flatten()
        
        # Bob measures
        if bob_bases[i] == 1:
            # Hadamard basis measurement
            h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            measure_state = h_gate @ state
        else:
            measure_state = state
            
        # Calculate measurement probabilities
        probs = np.array([abs(measure_state[0])**2, abs(measure_state[1])**2])
        # Normalize to handle numerical precision issues
        probs = probs / np.sum(probs)
        
        # Add some noise
        if np.random.random() < error_rate:
            bob_results.append(1 - np.random.choice([0, 1], p=probs))
        else:
            bob_results.append(np.random.choice([0, 1], p=probs))
    
    # Basis reconciliation
    matching_bases = alice_bases == bob_bases
    
    # Final key bits
    alice_key = alice_bits[matching_bases]
    bob_key = np.array(bob_results)[matching_bases]
    
    # Calculate error rate
    errors = np.sum(alice_key != bob_key)
    error_rate = errors / len(alice_key) if len(alice_key) > 0 else 0
    
    return alice_key, bob_key, error_rate


# %%
# Example 1: Ideal Conditions
# ---------------------------
#
# First, let's simulate the protocol under ideal conditions with no errors
# and no eavesdropper.

np.random.seed(42)
alice_key, bob_key, error_rate = run_bb84_protocol(
    num_qubits=100, 
    error_rate=0.0, 
    eavesdropper=False
)

print(f"Example 1 - Ideal conditions:")
print(f"First 10 bits of Alice's key: {alice_key[:10]}")
print(f"First 10 bits of Bob's key:   {bob_key[:10]}")
print(f"Error rate: {error_rate:.2%}")
print(f"Key length: {len(alice_key)} bits (from 100 transmitted qubits)")
print()

# %%
# Example 2: Noisy Channel
# ------------------------
#
# Now, let's simulate the protocol with some channel noise (5% error rate).

alice_key, bob_key, error_rate = run_bb84_protocol(
    num_qubits=100, 
    error_rate=0.05, 
    eavesdropper=False
)

print(f"Example 2 - Noisy channel (5%):")
print(f"First 10 bits of Alice's key: {alice_key[:10]}")
print(f"First 10 bits of Bob's key:   {bob_key[:10]}")
print(f"Error rate: {error_rate:.2%}")
print(f"Key length: {len(alice_key)} bits")
print()

# %%
# Example 3: Eavesdropper Present
# -------------------------------
#
# Finally, let's see what happens when Eve tries to intercept the communication.
# According to quantum mechanics, Eve's measurements will disturb the quantum states,
# introducing detectable errors.

alice_key, bob_key, error_rate = run_bb84_protocol(
    num_qubits=100, 
    error_rate=0.0, 
    eavesdropper=True
)

print(f"Example 3 - Eavesdropper present:")
print(f"First 10 bits of Alice's key: {alice_key[:10]}")
print(f"First 10 bits of Bob's key:   {bob_key[:10]}")
print(f"Error rate: {error_rate:.2%}")
print(f"Key length: {len(alice_key)} bits")
print()

# %%
# Comparing Error Rates Under Different Conditions
# ------------------------------------------------
#
# Let's run multiple simulations to compare error rates under different conditions.
# This demonstrates how eavesdropping introduces detectable errors.

np.random.seed(42)

# Run protocol with different settings
scenarios = [
    {"qubits": 1000, "error": 0.0, "eavesdrop": False, "label": "Ideal"},
    {"qubits": 1000, "error": 0.05, "eavesdrop": False, "label": "Noisy (5%)"},
    {"qubits": 1000, "error": 0.0, "eavesdrop": True, "label": "Eavesdropper"}
]

# Compare error rates between scenarios
print("\nComparing Error Rates Under Different Conditions:")
print("------------------------------------------------")
for scenario in scenarios:
    _, _, err_rate = run_bb84_protocol(
        scenario["qubits"], 
        scenario["error"], 
        scenario["eavesdrop"]
    )
    print(f"{scenario['label']} scenario: {err_rate:.2%} error rate")

# Key Retention Efficiency
# -----------------------
print("\nKey Retention Efficiency:")
print("------------------------")
qubits_range = [100, 500, 1000, 2000, 5000]
retained_ratios = []

for num_qubits in qubits_range:
    alice_key, _, _ = run_bb84_protocol(num_qubits, 0, False)
    retention = len(alice_key) / num_qubits
    retained_ratios.append(retention)
    print(f"{num_qubits} qubits: {retention:.2%} retention rate")

print(f"\nAverage retention rate: {np.mean(retained_ratios):.2%}")
print(f"Theoretical expectation: 50%")

# %%
# Conclusion
# ----------
#
# The BB84 protocol demonstrates how quantum mechanics can be used for secure key
# distribution. The key insights are:
#
# 1. **Information-theoretic security**: Unlike classical cryptography which relies on
#    computational hardness, quantum key distribution provides security based on
#    the laws of physics.
#
# 2. **Eavesdropper detection**: Any eavesdropping attempt introduces detectable errors
#    due to quantum measurement disturbance, typically around 25% error rate when Eve
#    measures each qubit.
#
# 3. **Key rate efficiency**: About 50% of transmitted qubits are retained as key bits
#    due to random basis selection, which is verified in our simulation.
#
# 4. **Practical considerations**: Real-world implementations must deal with noise,
#    decoherence, and imperfect equipment, which can reduce the effective key rate.
#
# The BB84 protocol has been implemented experimentally and commercially, with
# quantum key distribution systems capable of distributing keys over distances
# exceeding 100 kilometers through optical fibers or free space.