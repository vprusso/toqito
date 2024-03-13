"""Tests for is_unitary."""

import numpy as np

from toqito.channel_props import is_unitary
from toqito.channels import depolarizing
from toqito.perms import swap_operator


def test_is_unitary_identity_channel_true():
    """Verify that the identity channel is a unitary channel."""
    kraus_ops = [[np.identity(2), np.identity(2)]]

    np.testing.assert_equal(is_unitary(kraus_ops), True)


def test_is_unitary_depolarizing_false():
    """Verify that the Choi matrix of the depolarizing map is not a unitary channel."""
    np.testing.assert_equal(is_unitary(depolarizing(2)), False)


def test_is_unitary_isometry_false():
    """Verify that an isometry is not a unitary channel."""
    kraus_ops = [np.array([[1, 0, 0], [0, 1, 0]])]
    np.testing.assert_equal(is_unitary(kraus_ops), False)


def test_is_unitary_cp_channel_false():
    """Verify that a CP channel with two kraus ops is not a unitary channel."""
    kraus_ops = [np.identity(2), np.array([[0, 1], [1, 0]])]
    np.testing.assert_equal(is_unitary(kraus_ops), False)


def test_is_unitary_false():
    """Verify that a channel with one left and right kraus ops is not a unitary channel."""
    kraus_ops = [[np.identity(2), np.array([[0, 1], [1, 0]])]]
    np.testing.assert_equal(is_unitary(kraus_ops), False)


def test_is_unitary_transpose_map_false():
    """Verify that the channel that transposes 3x2 matrices is not unitary."""
    np.testing.assert_equal(is_unitary(swap_operator([2, 3])), False)
