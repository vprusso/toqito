"""Test brauer."""
import numpy as np

from toqito.states import brauer


def test_brauer_2_dim_2_pval():
    """Generate Brauer states on 4 qubits."""
    expected_res = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )

    res = brauer(2, 2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
