"""Channels is a module used to generate a number of widely-studied quantum channels."""

from toqito.channels.amplitude_damping import amplitude_damping
from toqito.channels.bitflip import bitflip
from toqito.channels.choi import choi
from toqito.channels.dephasing import dephasing
from toqito.channels.depolarizing import depolarizing
from toqito.channels.pauli_channel import pauli_channel
from toqito.channels.phase_damping import phase_damping
from toqito.channels.realignment import realignment
from toqito.channels.reduction import reduction  
from toqito.channel_ops import kraus_to_choi
from toqito.channel_ops import choi_to_kraus
from toqito.matrix_ops import partial_trace
from toqito.matrix_ops import partial_transpose
