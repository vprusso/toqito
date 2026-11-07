"""Channel properties module."""

from importlib import import_module
import sys
import types

__all__ = [
    "channel_dim",
    "choi_rank",
    "is_channel_antidistinguishable",
    "is_completely_positive",
    "is_extremal",
    "is_herm_preserving",
    "is_positive",
    "is_quantum_channel",
    "is_trace_preserving",
    "is_unital",
    "is_unitary",
    "is_unitary_antidistinguishable",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(f"{__name__}.{name}")
    attr = getattr(module, name)
    setattr(sys.modules[__name__], name, attr)
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)


class _ChannelPropsModule(types.ModuleType):
    """Keep the public functions from being replaced by their submodules.

    Each function lives in a submodule of the same name. Importing such a submodule directly (as other
    `toqito` modules do, e.g. `from toqito.channel_props.channel_dim import channel_dim`) makes Python
    set the submodule as an attribute of this package. That attribute would then hide the function, so
    `from toqito.channel_props import channel_dim` would return the module instead of the function.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if name in __all__ and isinstance(value, types.ModuleType):
            value = getattr(value, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ChannelPropsModule
