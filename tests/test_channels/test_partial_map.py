"""Tests for partial_map function."""
import unittest
import numpy as np

from toqito.channels import partial_map
from toqito.channels import depolarizing


class TestPartialMap(unittest.TestCase):
    """Unit test for partial_map."""

    def test_partial_map_depolarizing_first_system(self):
        """
        Perform the partial map using the depolarizing channel as the Choi
        matrix on first system.
        """
        rho = np.array(
            [
                [
                    0.3101,
                    -0.0220 - 0.0219 * 1j,
                    -0.0671 - 0.0030 * 1j,
                    -0.0170 - 0.0694 * 1j,
                ],
                [
                    -0.0220 + 0.0219 * 1j,
                    0.1008,
                    -0.0775 + 0.0492 * 1j,
                    -0.0613 + 0.0529 * 1j,
                ],
                [
                    -0.0671 + 0.0030 * 1j,
                    -0.0775 - 0.0492 * 1j,
                    0.1361,
                    0.0602 + 0.0062 * 1j,
                ],
                [
                    -0.0170 + 0.0694 * 1j,
                    -0.0613 - 0.0529 * 1j,
                    0.0602 - 0.0062 * 1j,
                    0.4530,
                ],
            ]
        )
        res = partial_map(rho, depolarizing(2))

        expected_res = np.array(
            [
                [0.20545 + 0.0j, 0.0 + 0.0j, -0.0642 + 0.02495j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.20545 + 0.0j, 0.0 + 0.0j, -0.0642 + 0.02495j],
                [-0.0642 - 0.02495j, 0.0 + 0.0j, 0.29455 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, -0.0642 - 0.02495j, 0.0 + 0.0j, 0.29455 + 0.0j],
            ]
        )

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_map_depolarizing_second_system(self):
        """
        Perform the partial map using the depolarizing channel as the Choi
        matrix on second system.
        """
        rho = np.array(
            [
                [
                    0.3101,
                    -0.0220 - 0.0219 * 1j,
                    -0.0671 - 0.0030 * 1j,
                    -0.0170 - 0.0694 * 1j,
                ],
                [
                    -0.0220 + 0.0219 * 1j,
                    0.1008,
                    -0.0775 + 0.0492 * 1j,
                    -0.0613 + 0.0529 * 1j,
                ],
                [
                    -0.0671 + 0.0030 * 1j,
                    -0.0775 - 0.0492 * 1j,
                    0.1361,
                    0.0602 + 0.0062 * 1j,
                ],
                [
                    -0.0170 + 0.0694 * 1j,
                    -0.0613 - 0.0529 * 1j,
                    0.0602 - 0.0062 * 1j,
                    0.4530,
                ],
            ]
        )
        res = partial_map(rho, depolarizing(2), 1)

        expected_res = np.array(
            [
                [0.2231 + 0.0j, 0.0191 - 0.00785j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0191 + 0.00785j, 0.2769 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.2231 + 0.0j, 0.0191 - 0.00785j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0191 + 0.00785j, 0.2769 + 0.0j],
            ]
        )

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partial_map_dim_list(self):
        """
        Perform the partial map using the depolarizing channel as the Choi
        matrix on first system when the dimension is specified as list.
        """
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        res = partial_map(rho, depolarizing(2), 2, [2, 2])

        expected_res = np.array(
            [
                [3.5, 0.0, 5.5, 0.0],
                [0.0, 3.5, 0.0, 5.5],
                [11.5, 0.0, 13.5, 0.0],
                [0.0, 11.5, 0.0, 13.5],
            ]
        )

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_non_square_matrix(self):
        """Matrix must be square."""
        with self.assertRaises(ValueError):
            rho = np.array([[1, 1, 1, 1], [5, 6, 7, 8], [3, 3, 3, 3]])
            partial_map(rho, depolarizing(3))

    def test_non_square_matrix_2(self):
        """Matrix must be square."""
        with self.assertRaises(ValueError):
            rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]])
            partial_map(rho, depolarizing(3), 2)

    def test_invalid_dim(self):
        """Invalid dimension for partial map."""
        with self.assertRaises(ValueError):
            rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]])
            partial_map(rho, depolarizing(3), 1, [2, 2])


if __name__ == "__main__":
    unittest.main()
