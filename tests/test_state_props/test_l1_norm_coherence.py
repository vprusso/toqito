"""Test l1_norm_coherence."""
import numpy as np

from toqito.state_props import l1_norm_coherence


def test_l1_norm_coherence_maximally_coherence():
    """The l1-norm coherence of the maximally coherent state."""
    v_vec = np.ones((3, 1)) / np.sqrt(3)
    res = l1_norm_coherence(v_vec)

    np.testing.assert_equal(np.isclose(res, 2), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
