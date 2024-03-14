"""Tests for kraus_to_choi."""

import numpy as np
import pytest

from toqito.channel_ops import kraus_to_choi
from toqito.channels import dephasing, depolarizing

kraus_1_transpose = np.array([[1, 0], [0, 0]])
kraus_2_transpose = np.array([[1, 0], [0, 0]]).conj().T
kraus_3_transpose = np.array([[0, 1], [0, 0]])
kraus_4_transpose = np.array([[0, 1], [0, 0]]).conj().T
kraus_5_transpose = np.array([[0, 0], [1, 0]])
kraus_6_transpose = np.array([[0, 0], [1, 0]]).conj().T
kraus_7_transpose = np.array([[0, 0], [0, 1]])
kraus_8_transpose = np.array([[0, 0], [0, 1]]).conj().T

kraus_ops_transpose = [
    [kraus_1_transpose, kraus_2_transpose],
    [kraus_3_transpose, kraus_4_transpose],
    [kraus_5_transpose, kraus_6_transpose],
    [kraus_7_transpose, kraus_8_transpose],
]


expected_choi_res_transpose = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

kraus_1_swap_operator_non_unique = np.array([[0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]])
kraus_2_swap_operator_non_unique = np.array([[0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]]).conj().T
kraus_3_swap_operator_non_unique = np.array([[1, 0], [0, 0]])
kraus_4_swap_operator_non_unique = np.array([[1, 0], [0, 0]]).conj().T
kraus_5_swap_operator_non_unique = np.array([[0, 0], [0, 1]])
kraus_6_swap_operator_non_unique = np.array([[0, 0], [0, 1]]).conj().T
kraus_7_swap_operator_non_unique = np.array([[0, 1 / np.sqrt(2)], [-1 / np.sqrt(2), 0]])
kraus_8_swap_operator_non_unique = np.array([[0, 1 / np.sqrt(2)], [-1 / np.sqrt(2), 0]]).conj().T

kraus_ops_swap_operator_non_unique = [
    [kraus_1_swap_operator_non_unique, kraus_2_swap_operator_non_unique],
    [kraus_3_swap_operator_non_unique, kraus_4_swap_operator_non_unique],
    [kraus_5_swap_operator_non_unique, kraus_6_swap_operator_non_unique],
    [kraus_7_swap_operator_non_unique, kraus_8_swap_operator_non_unique],
]

expected_choi_res_swap_operator_non_unique = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

kraus_1_dephasing_channel = np.array([[1, 0], [0, 0]])
kraus_2_dephasing_channel = np.array([[1, 0], [0, 0]])
kraus_3_dephasing_channel = np.array([[0, 0], [0, 1]])
kraus_4_dephasing_channel = np.array([[0, 0], [0, 1]])

kraus_ops_dephasing_channel = [
    [kraus_1_dephasing_channel, kraus_2_dephasing_channel],
    [kraus_3_dephasing_channel, kraus_4_dephasing_channel],
]

expected_choi_res_dephasing_channel = dephasing(2)


kraus_1_depolarizing_channel = np.array([[1 / np.sqrt(2), 0], [0, 0]])
kraus_2_depolarizing_channel = np.array([[1 / np.sqrt(2), 0], [0, 0]])
kraus_3_depolarizing_channel = np.array([[0, 0], [1 / np.sqrt(2), 0]])
kraus_4_depolarizing_channel = np.array([[0, 0], [1 / np.sqrt(2), 0]])
kraus_5_depolarizing_channel = np.array([[0, 1 / np.sqrt(2)], [0, 0]])
kraus_6_depolarizing_channel = np.array([[0, 1 / np.sqrt(2)], [0, 0]])
kraus_7_depolarizing_channel = np.array([[0, 0], [0, 1 / np.sqrt(2)]])
kraus_8_depolarizing_channel = np.array([[0, 0], [0, 1 / np.sqrt(2)]])

kraus_ops_depolarizing_channel = [
    [kraus_1_depolarizing_channel, kraus_2_depolarizing_channel],
    [kraus_3_depolarizing_channel, kraus_4_depolarizing_channel],
    [kraus_5_depolarizing_channel, kraus_6_depolarizing_channel],
    [kraus_7_depolarizing_channel, kraus_8_depolarizing_channel],
]
expected_choi_res_depolarizing_channel = depolarizing(2)

v_mat = np.array([[1, 0, 0], [0, 1, 0]])
expected_v_mat = np.array(
    [
        [1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

kraus_1 = np.array([[1, 0, 0], [0, 1, 0]])

kraus_2 = np.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
    ]
)

expected_non_square = np.array(
    [
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # Choi matrix of the transpose map is the swap operator.
        (kraus_ops_transpose, expected_choi_res_transpose),
        # As Kraus operators are non-unique, these also should yield the swap operator
        (kraus_ops_swap_operator_non_unique, expected_choi_res_swap_operator_non_unique),
        # Kraus operators for dephasing channel should yield the proper Choi matrix.
        (kraus_ops_dephasing_channel, expected_choi_res_dephasing_channel),
        # Kraus operators for depolarizing channel should yield the proper Choi matrix
        (kraus_ops_depolarizing_channel, expected_choi_res_depolarizing_channel),
        # Kraus operators for an isometry
        ([v_mat], expected_v_mat),
        # Kraus operators for non square inputs and outputs
        ([[kraus_1, kraus_2]], expected_non_square),
    ],
)
def test_kraus_to_choi(test_input, expected):
    """Test function works as expected for valid inputs."""
    calculated = kraus_to_choi(test_input)
    assert np.isclose(calculated, expected).all()
