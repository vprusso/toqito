"""Optimizations over quantum channels: a set of modules implementing SDP optimization problems for quantum channels.

These optimizations over quantum channels return optimal results. They are listed below.
"""

from toqito.channel_opt.channel_distinguishability import channel_distinguishability
from toqito.channel_opt.channel_exclusion import channel_exclusion
from toqito.channel_opt.unitary_exclusion import unitary_exclusion
