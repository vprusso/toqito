"""CVXQUAD-derived matrix cones and entropy functionals for use in CVXPY SDPs.

This package collects operator-monotone matrix functions and the associated
SDP cones (matrix geometric mean, operator relative entropy, matrix logarithm)
ported from CVXQUAD (Fawzi--Saunderson). They live here rather than in
``matrix_ops`` because they depend on ``matrix_props`` for input validation;
keeping ``matrix_ops`` free of that back-edge avoids a load-time import cycle
with ``state_props`` and ``channels``.
"""

from toqito.cones.geometric_mean import geometric_mean
from toqito.cones.geometric_mean_epi_cone import geometric_mean_epi_cone
from toqito.cones.geometric_mean_hypo_cone import geometric_mean_hypo_cone
from toqito.cones.lieb_ando import lieb_ando
from toqito.cones.ln_quantum_entropy import ln_quantum_entropy
from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)
from toqito.cones.trace_matrix_log import trace_matrix_log
from toqito.cones.trace_matrix_power import trace_matrix_power
