"""Test unique_perms."""
import numpy as np

from toqito.perms import unique_perms


def test_unique_perms_len():
    """Checks the number of unique perms."""
    vec = [1, 1, 2, 2, 1, 2, 1, 3, 3, 3]
    np.testing.assert_equal(len(list(unique_perms(vec))), 4200)


if __name__ == "__main__":
    np.testing.run_module_suite()
