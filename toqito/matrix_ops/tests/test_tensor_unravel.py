import unittest
import numpy as np

from tensor_unravel import tensor_unravel

class TestTensorUnravel(unittest.TestCase):
    
    def test_tensor_unravel_basic(self):
        """2D tensor with one positive element at (1,1)."""
        
        tensor_constraint = np.array([[-1, -1], [-1, 1]])
        expected = np.array([1, 1, 1])
        result = tensor_unravel(tensor_constraint)
        np.testing.assert_array_equal(result, expected)
    
    def test_tensor_unravel_diagonal_unique(self):
        """3D tensor with the unique positive element at (1,1,1)."""
        
        tensor_constraint = np.full((2, 2, 2), -1)
        tensor_constraint[1, 1, 1] = 1
        expected = np.array([1, 1, 1, 1])
        result = tensor_unravel(tensor_constraint)
        np.testing.assert_array_equal(result, expected)
    
    def test_tensor_unravel_invalid_tensor(self):
        """Raise ValueError if no unique positive element exists."""
        
        tensor_constraint = np.full((2, 2), -1)
        with self.assertRaises(ValueError) as context:
            tensor_unravel(tensor_constraint)
        self.assertIn("does not have exactly two distinct values", str(context.exception))
    
    def test_tensor_unravel_multiple_unique_elements(self):
        """Raise ValueError if multiple unique positive elements exist."""
        
        tensor_constraint = np.full((2, 2, 2), -1)
        tensor_constraint[0, 0, 0] = 1
        tensor_constraint[1, 1, 1] = 1
        with self.assertRaises(ValueError) as context:
            tensor_unravel(tensor_constraint)
        self.assertIn("unique element", str(context.exception))

