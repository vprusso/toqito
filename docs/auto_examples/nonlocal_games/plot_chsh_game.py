"""
CHSH Nonlocal Game
==================

This example demonstrates the CHSH nonlocal game using toqito.
"""
# sphinx_gallery_thumbnail_number = 1  # Use the first figure as thumbnail


import numpy as np
import matplotlib.pyplot as plt
from toqito.nonlocal_games.xor_game import XORGame

# The probability matrix
prob_mat = np.array([[1/4, 1/4], [1/4, 1/4]])

# The predicate matrix
pred_mat = np.array([[0, 0], [0, 1]])

# Define CHSH game from matrices
chsh = XORGame(prob_mat, pred_mat)

# Calculate values
classical_val = chsh.classical_value()  # 0.75
quantum_val = chsh.quantum_value()  # 0.8535533

# Plot the results
plt.figure(figsize=(8, 6))
strategies = ['Classical', 'Quantum']
values = [classical_val, quantum_val]
plt.bar(strategies, values, color=['blue', 'purple'])
plt.ylim(0, 1)
plt.title('CHSH Game: Classical vs Quantum Strategy')
plt.ylabel('Success Probability')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

plt.show()
