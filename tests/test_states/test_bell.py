"""Test bell."""
import numpy as np

from toqito.states import basis
from toqito.states import bell


def test_bell_0():
    """Generate the Bell state: `1/sqrt(2) * (|00> + |11>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))

    res = bell(0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_1():
    """Generates the Bell state: `1/sqrt(2) * (|00> - |11>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1))

    res = bell(1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_2():
    """Generates the Bell state: `1/sqrt(2) * (|01> + |10>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0))

    res = bell(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_3():
    """Generates the Bell state: `1/sqrt(2) * (|01> - |10>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))

    res = bell(3)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_invalid():
    """Ensures that an integer above 3 is error-checked."""
    with np.testing.assert_raises(ValueError):
        bell(4)


if __name__ == "__main__":
    np.testing.run_module_suite()
