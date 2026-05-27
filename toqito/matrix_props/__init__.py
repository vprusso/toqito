"""Matrix operations is a set of modules that implements various properties of matrices and vectors."""

import sys
from importlib import import_module

from toqito.matrix_props.has_same_dimension import has_same_dimension
from toqito.matrix_props.is_square import is_square
from toqito.matrix_props.kp_norm import kp_norm
from toqito.matrix_props.is_anti_hermitian import is_anti_hermitian
from toqito.matrix_props.is_hermitian import is_hermitian
from toqito.matrix_props.is_positive_semidefinite import is_positive_semidefinite
from toqito.matrix_props.factor_width import factor_width
from toqito.matrix_props.is_rank_one import is_rank_one
from toqito.matrix_props.is_density import is_density
from toqito.matrix_props.is_circulant import is_circulant
from toqito.matrix_props.is_diagonal import is_diagonal
from toqito.matrix_props.is_normal import is_normal
from toqito.matrix_props.is_orthonormal import is_orthonormal
from toqito.matrix_props.is_symmetric import is_symmetric
from toqito.matrix_props.is_identity import is_identity
from toqito.matrix_props.is_idempotent import is_idempotent
from toqito.matrix_props.is_permutation import is_permutation
from toqito.matrix_props.is_positive_definite import is_positive_definite
from toqito.matrix_props.is_commuting import is_commuting
from toqito.matrix_props.is_projection import is_projection
from toqito.matrix_props.is_unitary import is_unitary
from toqito.matrix_props.majorizes import majorizes
# sk_operator_norm is loaded lazily below: sk_norm transitively imports state_props
# (for schmidt_rank / sk_vector_norm), and several state_props modules import back
# into matrix_ops, which creates a load-time cycle for any matrix_ops module that
# imports matrix_props at top level. Deferring this single attribute keeps every
# other matrix_props consumer free to do plain top-level imports.
from toqito.matrix_props.is_block_positive import is_block_positive
from toqito.matrix_props.trace_norm import trace_norm
from toqito.matrix_props.is_diagonally_dominant import is_diagonally_dominant
from toqito.matrix_props.is_totally_positive import is_totally_positive
from toqito.matrix_props.is_linearly_independent import is_linearly_independent
from toqito.matrix_props.is_nonnegative import is_nonnegative
from toqito.matrix_props.is_positive import is_positive
from toqito.matrix_props.positive_semidefinite_rank import positive_semidefinite_rank
from toqito.matrix_props.is_stochastic import is_stochastic
from toqito.matrix_props.spark import spark
from toqito.matrix_props.is_pseudo_unitary import is_pseudo_unitary
from toqito.matrix_props.is_pseudo_hermitian import is_pseudo_hermitian
from toqito.matrix_props.commutant import commutant
from toqito.matrix_props.mutual_coherence import mutual_coherence
from toqito.matrix_props.is_absolutely_k_incoherent import is_absolutely_k_incoherent
from toqito.matrix_props.is_k_incoherent import is_k_incoherent
from toqito.matrix_props.is_ldoi import is_ldoi
from toqito.matrix_props.is_tight_frame import is_tight_frame
from toqito.matrix_props.is_equiangular_tight_frame import is_equiangular_tight_frame
from toqito.matrix_props.nonnegative_rank import nonnegative_rank

_LAZY_ATTRS = {"sk_operator_norm": "sk_norm"}


def __getattr__(name: str) -> object:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(f"{__name__}.{_LAZY_ATTRS[name]}")
    attr = getattr(module, name)
    setattr(sys.modules[__name__], name, attr)
    return attr
