from toqito.helper.unique_perms import unique_perms

import unittest


class TestWState(unittest.TestCase):
    """Unit test for w_state."""

    def test_unique_perms_len(self):
        """Checks the number of unique perms"""
        v = [1, 1, 2, 2, 1, 2, 1, 3, 3, 3]

        self.assertEqual(len(list(unique_perms(v))), 4200)


if __name__ == '__main__':
    unittest.main()
