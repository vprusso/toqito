"""Test has_symmetric_inner_extension."""

import numpy as np

from toqito.state_props import has_symmetric_inner_extension
from toqito.states import max_entangled


def test_has_symmetric_inner_extension_qetlab_example():
    """QETLAB example: a specific 3x3 PPT inner extension certifies separability."""
    rho = np.array(
        [
            [11, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 11, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 11, -1, -1, 4, -1, -1, -1],
            [-1, -1, -1, 11, -1, -1, 4, -1, -1],
            [-1, -1, -1, -1, 16, -1, -1, -1, -1],
            [-1, -1, 4, -1, -1, 11, -1, -1, -1],
            [-1, -1, -1, 4, -1, -1, 11, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 11, 4],
            [-1, -1, -1, -1, -1, -1, -1, 4, 11],
        ],
        dtype=float,
    )

    assert has_symmetric_inner_extension(rho, level=2, dim=[3, 3], ppt=True)


def test_has_symmetric_inner_extension_rejects_maximally_entangled_qutrit():
    """A maximally entangled qutrit state is not certified by the inner cone."""
    psi = max_entangled(3)
    rho = psi @ psi.conj().T

    assert not has_symmetric_inner_extension(rho, level=2, dim=[3, 3], ppt=True)
