"""Test random_povm."""
import numpy as np

from toqito.random import random_povm


def test_random_povm_unitary_not_real():
    """Generate random POVMs and check that they sum to the identity."""
    dim, num_inputs, num_outputs = 2, 2, 2
    povms = random_povm(dim, num_inputs, num_outputs)
    np.testing.assert_equal(
        np.allclose(povms[:, :, 0, 0] + povms[:, :, 0, 1], np.identity(dim)), True
    )


if __name__ == "__main__":
    np.testing.run_module_suite()
