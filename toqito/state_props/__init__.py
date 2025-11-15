"""State properties module."""

from importlib import import_module
import sys

__all__ = [
    "abs_ppt_constraints",
    "bures_angle",
    "bures_distance",
    "common_quantum_overlap",
    "concurrence",
    "entanglement_of_formation",
    "fidelity",
    "fidelity_of_separability",
    "has_symmetric_extension",
    "helstrom_holevo",
    "hilbert_schmidt",
    "hilbert_schmidt_inner_product",
    "in_separable_ball",
    "is_abs_ppt",
    "is_antidistinguishable",
    "is_distinguishable",
    "is_ensemble",
    "is_mixed",
    "is_mutually_orthogonal",
    "is_mutually_unbiased_basis",
    "is_npt",
    "is_product",
    "is_ppt",
    "is_pure",
    "is_separable",
    "is_unextendible_product_basis",
    "l1_norm_coherence",
    "learnability",
    "log_negativity",
    "matsumoto_fidelity",
    "negativity",
    "purity",
    "renyi_entropy",
    "schmidt_rank",
    "sk_vector_norm",
    "sub_fidelity",
    "trace_distance",
    "von_neumann_entropy",
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
