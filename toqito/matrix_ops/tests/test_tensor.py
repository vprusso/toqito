"""Test tensor."""
import numpy as np
import pytest

from toqito.matrix_ops import tensor


@pytest.mark.parametrize("input_args, expected_output", [
    # Test with two matrices.
    ((np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])), 
     np.array([[ 5,  6, 10, 12],
               [ 7,  8, 14, 16],
               [15, 18, 20, 24],
               [21, 24, 28, 32]])),

    # Test with three matrices.
    ((np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])), 
        np.array([[45, 50, 54, 60, 90, 100, 108, 120],
               [55, 60, 66, 72, 110, 120, 132, 144],
               [63, 70, 72, 80, 126, 140, 144, 160],
               [77, 84, 88, 96, 154, 168, 176, 192],
               [135, 150, 162, 180, 180, 200, 216, 240],
               [165, 180, 198, 216, 220, 240, 264, 288],
               [189, 210, 216, 240, 252, 280, 288, 320],
               [231, 252, 264, 288, 308, 336, 352, 384]])),

    # Test with multiple matrices.
    ((np.identity(2), np.identity(2), np.identity(2), np.identity(2)), np.identity(16)),

    # Test with a matrix repeated 0 times should return None.
    ((np.array([[1, 2], [3, 4]]), 0), None),

    # Test with a matrix repeated 1 times.
    ((np.array([[1, 2], [3, 4]]), 1), 
     np.array([[1, 2],
               [3, 4]])),

    # Test with a matrix repeated 2 times.
    ((np.array([[1, 2], [3, 4]]), 2), 
     np.array([[ 1,  2,  2,  4],
               [ 3,  4,  6,  8],
               [ 3,  6,  4,  8],
               [ 9, 12, 12, 16]])),

    # Test with a matrix repeated 3 times.
    ((np.array([[1, 2], [3, 4]]), 3), 
     np.array([[1, 2, 2, 4, 2, 4, 4, 8],
               [3, 4, 6, 8, 6, 8, 12, 16],
               [3, 6, 4, 8, 6, 12, 8, 16],
               [9, 12, 12, 16, 18, 24, 24, 32],
               [3, 6, 6, 12, 4, 8, 8, 16],
               [9, 12, 18, 24, 12, 16, 24, 32],
               [9, 18, 12, 24, 12, 24, 16, 32],
               [27, 36, 36, 48, 36, 48, 48, 64]])),

    # Test with standard basis: |0> \otimes |0>:
    ((np.array([[1], [0]]), np.array([[1], [0]])), np.array([[1], [0], [0], [0]])),

    # Performing tensor product on one item should return item back.
    ((np.array([[1, 2], [3, 4]]),), np.array([[1, 2], [3, 4]])),

    # Test with numpy array list of one element.
    (np.array([np.array([[1, 2], [3, 4]])]), 
     np.array([[1, 2],
               [3, 4]])),

    # Test with a list containing a single matrix
    (([np.array([[1, 2], [3, 4]])],), np.array([[1, 2], [3, 4]])),

    # Test with a list containing exactly two matrices
    (([np.array([[1, 2]]), np.array([[3, 4]])],), np.array([[3, 4, 6, 8]])),

    # Test with a list containing three matrices
    (([np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5, 6]])],),
     np.array([[15, 18, 20, 24, 30, 36, 40, 48]])),

    # Test with numpy array list of two elements.
    ([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])], 
     np.array([[ 5,  6, 10, 12],
               [ 7,  8, 14, 16],
               [15, 18, 20, 24],
               [21, 24, 28, 32]])),

    # Test with numpy array list of three elements.
    ([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])], 
        np.array([[45, 50, 54, 60, 90, 100, 108, 120],
               [55, 60, 66, 72, 110, 120, 132, 144],
               [63, 70, 72, 80, 126, 140, 144, 160],
               [77, 84, 88, 96, 154, 168, 176, 192],
               [135, 150, 162, 180, 180, 200, 216, 240],
               [165, 180, 198, 216, 220, 240, 264, 288],
               [189, 210, 216, 240, 252, 280, 288, 320],
               [231, 252, 264, 288, 308, 336, 352, 384]])),

    # Test with list of multiple elements.
    ([np.identity(2), np.identity(2), np.identity(2), np.identity(2)], np.identity(16)),

    # Test with an empty list should return None.
    (([], ), None),

    # Test tensor list with one item.
    ([np.identity(2)], np.identity(2)),

    # Test tensor list with two items.
    ([np.identity(2), np.identity(2)], np.identity(4)),

    # Test tensor list with three items.
    ([np.identity(2), np.identity(2), np.identity(2)], np.identity(8)),

    # Test with a single 2D numpy array (matrix)
    ((np.array([[1, 2], [3, 4]]),), np.array([[1, 2], [3, 4]])),

    # Test with a numpy array containing two matrices (or vectors)
    ((np.array([np.array([1, 2]), np.array([3, 4])])), 
     np.array([3, 4, 6, 8])),

    # Test with a numpy array containing three matrices (or vectors)
    ((np.array([np.array([1, 2]), np.array([3, 4]), np.array([5, 6])])), 
    np.array([15, 18, 20, 24, 30, 36, 40, 48])),
])
def test_tensor(input_args, expected_output):
    result = tensor(*input_args)
    np.testing.assert_array_equal(result, expected_output)


def test_tensor_empty_args():
    """Test tensor with no arguments."""
    with np.testing.assert_raises(ValueError):
        tensor()
