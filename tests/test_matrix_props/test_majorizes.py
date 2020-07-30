"""Test majorizes."""
import numpy as np

from toqito.channels import partial_trace
from toqito.matrix_props import majorizes
from toqito.states import max_entangled


def test_majorizes_simple_example():
    """Test that simple example of vectors returns True."""
    np.testing.assert_equal(majorizes([3, 0, 0], [1, 1, 1]), True)


def test_majorizes_max_entangled():
    """Test that max entangled partial trace returns False."""
    v_vec = max_entangled(3)
    rho = v_vec * v_vec.conj().T
    np.testing.assert_equal(majorizes(partial_trace(rho), rho), False)


def test_majorizes_max_entangled_flip():
    """Test that max entangled partial trace returns True (flipped args)."""
    v_vec = max_entangled(3)
    rho = v_vec * v_vec.conj().T
    np.testing.assert_equal(majorizes(rho, partial_trace(rho)), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
