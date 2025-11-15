"""Channel properties module."""

from importlib import import_module
import sys

__all__ = [
    "channel_dim",
    "channel_distinguishability",
    "channel_fidelity",
    "choi_rank",
    "completely_bounded_spectral_norm",
    "completely_bounded_trace_norm",
    "diamond_distance",
    "fidelity_of_separability",
    "is_completely_positive",
    "is_extremal",
    "is_herm_preserving",
    "is_positive",
    "is_quantum_channel",
    "is_trace_preserving",
    "is_unital",
    "is_unitary",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(f"{__name__}.{name}")
    attr = getattr(module, name)
    setattr(sys.modules[__name__], name, attr)
    return attr


def __dir__():
    return sorted(__all__)
