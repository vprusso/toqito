"""Test is_unextendible_product_basis."""

import numpy as np
import pytest

from toqito.state_props.is_unextendible_product_basis import (
    is_unextendible_product_basis,
)
from toqito.matrix_ops import tensor
from toqito.matrix_ops import inner_product


def Tiles():
    """Constructs Tiles UPB."""
    tiles_A = np.zeros([3, 5])
    tiles_A[:, 0] = [1, 0, 0]
    tiles_A[:, 1] = [1 / np.sqrt(2), -1 / np.sqrt(2), 0]
    tiles_A[:, 2] = [0, 0, 1]
    tiles_A[:, 3] = [0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
    tiles_A[:, 4] = [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]

    tiles_B = np.zeros([3, 5])
    tiles_B[:, 0] = [1 / np.sqrt(2), -1 / np.sqrt(2), 0]
    tiles_B[:, 1] = [0, 0, 1]
    tiles_B[:, 2] = [0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
    tiles_B[:, 3] = [1, 0, 0]
    tiles_B[:, 4] = [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]

    return [tiles_A, tiles_B]


def GenShifts(parties: int):
    """Constructs GenShifts UPB for odd number of parties."""
    if (parties % 2 == 0) or parties == 1:
        raise ValueError("Input must be an odd int greater than 1.")

    k = int((parties + 1) / 2)
    num_states = 2 * k
    upb = [np.zeros([2, num_states]) for _ in range(parties)]
    angles = [(i / k) * np.pi / 2 for i in range(2 * k - 1, k, -1)]
    local_states = [[np.cos(angle), np.sin(angle)] for angle in angles]
    orth_angles = [(i / k) * np.pi / 2 for i in range(1, k)]
    orth_local_states = [[np.cos(angle), np.sin(angle)] for angle in orth_angles]

    state_list = []
    state_list.append([0, 1])
    state_list.extend(local_states)
    state_list.extend(orth_local_states)

    for party in upb:
        party[:, 0] = [1, 0]

    for state in range(1, num_states):
        for i, party in enumerate(upb):
            party[:, state] = state_list[i]
        state_list = state_list[-1:] + state_list[:-1]

    return upb


def test_is_unextendible_product_basis_input_empty_list():
    """Empty list as input."""
    with np.testing.assert_raises(ValueError):
        empty_list = []
        is_unextendible_product_basis(empty_list)


def test_is_unextendible_product_basis_input_not_numpy_arrays():
    """List elements are not type numpy.ndarray."""
    with np.testing.assert_raises(ValueError):
        list_of_listarrays = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        is_unextendible_product_basis(list_of_listarrays)


def test_is_unextendible_product_basis_input_arrays_not_two_dimensional():
    """Arrays are not two-dimensional."""
    with np.testing.assert_raises(ValueError):
        array_1D = np.array([1, 2, 3, 4])
        input = [array_1D, array_1D]
        is_unextendible_product_basis(input)


def test_is_unextendible_product_basis_input_arrays_not_same_num_columns():
    """Arrays do not have the same number of columns"""
    with np.testing.assert_raises(ValueError):
        array_1 = np.array([[1, 2], [3, 4]])
        array_2 = np.array([[1], [3]])
        input = [array_1, array_2]
        is_unextendible_product_basis(input)


def test_is_unextendable_product_basis_tiles():
    """Verify if Tiles UPB is a UPB"""
    res = is_unextendible_product_basis(Tiles())
    expected_res = [True, None]
    np.testing.assert_array_equal(res, expected_res)


@pytest.mark.parametrize("num_states", [1, 2, 3, 4])
def test_is_unextendable_product_basis_tiles_remove_states_false(num_states):
    """Check if Tiles UPB fails to be a UPB when 1, 2, 3, and 4 states are removed."""
    tiles = Tiles()
    res = is_unextendible_product_basis(
        [tiles[0][0:, 0:num_states], tiles[1][:, 0:num_states]]
    )
    expected_res = False
    np.testing.assert_equal(res[0], expected_res)


@pytest.mark.parametrize("num_states", [1, 2, 3, 4])
def test_is_unextendable_product_basis_tiles_remove_states_orthogonal_witness(
    num_states,
):
    """Check if witness is orthogonal to Tiles UPB when 1, 2, 3, and 4 states are removed."""
    tiles = Tiles()
    witness = is_unextendible_product_basis(
        [tiles[0][:, 0:num_states], tiles[1][:, 0:num_states]]
    )[1]
    witness_product = tensor(witness[0], witness[1])
    # Perhaps we can use `toqito.state_props.is_mutually_orthogonal` to test orthogonallity
    inner_product_list = []
    for i in range(num_states):
        UPB_state_product = tensor(
            np.array([tiles[0][:, i]]).T, np.array([tiles[1][:, i]]).T
        )
        ip = inner_product(witness_product[:, 0], UPB_state_product[:, 0])
        inner_product_list.append(ip)
    res = inner_product_list
    expected_res = [0] * num_states
    np.testing.assert_array_almost_equal(res, expected_res)


@pytest.mark.parametrize("num_parties", [3, 5, 7])
def test_is_unextendable_product_basis_tiles_GenShifts(num_parties):
    """Verify if GenShifts UPB is a UPB for 3, 5, and, 7 parties"""
    res = is_unextendible_product_basis(GenShifts(num_parties))
    expected_res = [True, None]
    np.testing.assert_array_equal(res, expected_res)


if __name__ == "__main__":
    np.testing.run_module_suite()
