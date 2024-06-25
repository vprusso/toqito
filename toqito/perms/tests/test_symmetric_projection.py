"""Test symmetric_projection."""

import numpy as np

from toqito.perms import symmetric_projection


def test_symmetric_projection_dim_2_pval_1():
    """Symmetric_projection where the dimension is 2 and p_val is 1."""
    res = symmetric_projection(dim=2, p_val=1)
    expected_res = np.array([[1, 0], [0, 1]])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_2_pval_default_2():
    """Generates the symmetric_projection where the dimension is 2."""
    res = symmetric_projection(dim=2)
    expected_res = np.array([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


# def test_symmetric_projection_dim_2_pval_2_partial_true():
#     """Generates the symmetric_projection where the dimension is 2 and partial is True."""
#     res = symmetric_projection(dim=2, p_val=2, partial=True)
#     expected_res = np.array([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]])
#
#     bool_mat = np.isclose(res, expected_res)
#     np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_2_partial_true():
    """Symmetric_projection where the dimension is 2 and partial is True."""
    res = symmetric_projection(dim=2, p_val=2, partial=True)
    expected_res = symmetric_projection(dim=2, p_val=2, partial=False)

    bool_mat = np.isclose(res @ res.conj().T, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_2_pval_3():
    """Symmetric_projection where the dimension is 2 and p_val is 3."""
    res = symmetric_projection(dim=2, p_val=3)
    expected_res = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0],
            [0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0],
            [0, 0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0],
            [0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0],
            [0, 0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0],
            [0, 0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_3_pval_2():
    """Symmetric_projection where the dimension is 3 and p_val is 2."""
    res = symmetric_projection(dim=3, p_val=2)
    expected_res = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1 / 2, 0, 1 / 2, 0, 0, 0, 0, 0],
            [0, 0, 1 / 2, 0, 0, 0, 1 / 2, 0, 0],
            [0, 1 / 2, 0, 1 / 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1 / 2, 0, 1 / 2, 0],
            [0, 0, 1 / 2, 0, 0, 0, 1 / 2, 0, 0],
            [0, 0, 0, 0, 0, 1 / 2, 0, 1 / 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_4_pval_2():
    """Symmetric_projection where the dimension is 4 and p_val is 2."""
    res = symmetric_projection(dim=4, p_val=2)
    expected_res = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1 / 2, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1 / 2, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 1 / 2, 0, 0],
            [0, 0, 1 / 2, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 1 / 2, 0],
            [0, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0, 0, 0, 1 / 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 1 / 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_4_pval_2_partial_true():
    """Dimension is 4, p_val is 2, and partial is True."""
    res = symmetric_projection(dim=4, p_val=2, partial=True)
    expected_res = symmetric_projection(dim=4, p_val=2, partial=False)

    bool_mat = np.isclose(res @ res.conj().T, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_invalid_dim():
    """Test for invalid dimension."""
    try:
        symmetric_projection(dim=0, p_val=2)
    except ValueError as e:
        assert str(e) == "InvalidDim: `dim` must be at least 1."


def test_symmetric_projection_invalid_pval():
    """Test for invalid p_val."""
    try:
        symmetric_projection(dim=2, p_val=0)
    except ValueError as e:
        assert str(e) == "InvalidPVal: `p_val` must be at least 1."
