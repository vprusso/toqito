"""Channels is a module used to generate a number of widely-studied quantum channels."""

# Imports for modules/functions defined *within* the toqito/channels/ directory
from toqito.channels.amplitude_damping import amplitude_damping
from toqito.channels.bitflip import bitflip
from toqito.channels.choi import choi # Assuming choi.py is in toqito/channels/
from toqito.channels.dephasing import dephasing
from toqito.channels.depolarizing import depolarizing
from toqito.channels.pauli_channel import pauli_channel
from toqito.channels.phase_damping import phase_damping
from toqito.channels.realignment import realignment # Assuming realignment.py is in toqito/channels/
from toqito.channels.reduction import reduction   # Assuming reduction.py is in toqito/channels/ (add if needed)

# Re-exports from other toqito subpackages to make them available under toqito.channels
# These should now be safe after foundational utilities were moved.
from toqito.channel_ops import kraus_to_choi
from toqito.channel_ops import choi_to_kraus

# Optional: Re-exports for backward compatibility if users expect these from toqito.channels
# Internally, toqito should prefer `from toqito.matrix_ops import ...`
from toqito.matrix_ops import partial_trace
from toqito.matrix_ops import partial_transpose

# Define the public API of this module
__all__ = ["amplitude_damping", "bitflip", "choi", "choi_to_kraus", "dephasing",
    "depolarizing", "kraus_to_choi", "partial_trace", "partial_transpose",
    "pauli_channel", "phase_damping", "realignment", "reduction",
]
