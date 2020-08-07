"""Test state_exclusion."""
import numpy as np

from toqito.state_opt import state_exclusion
from toqito.states import bell


def test_conclusive_state_exclusion_one_state():
    """Conclusive state exclusion for single state."""
    rho = bell(0) * bell(0).conj().T
    states = [rho]

    res = state_exclusion(states, probs=None, method="conclusive")
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_conclusive_state_exclusion_one_state_vec():
    """Conclusive state exclusion for single vector state."""
    rho = bell(0)
    states = [rho]

    res = state_exclusion(states, probs=None, method="conclusive")
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_conclusive_state_exclusion_three_state():
    """Conclusive state exclusion for three Bell state density matrices."""
    rho1 = bell(0) * bell(0).conj().T
    rho2 = bell(1) * bell(1).conj().T
    rho3 = bell(2) * bell(2).conj().T
    states = [rho1, rho2, rho3]
    probs = [1 / 3, 1 / 3, 1 / 3]

    res = state_exclusion(states, probs, "conclusive")
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_conclusive_state_exclusion_three_state_vec():
    """Conclusive state exclusion for three Bell state vectors."""
    rho1 = bell(0)
    rho2 = bell(1)
    rho3 = bell(2)
    states = [rho1, rho2, rho3]
    probs = [1 / 3, 1 / 3, 1 / 3]

    res = state_exclusion(states, probs, "conclusive")
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_conclusive_state_exclusion_complex_three_state_vec():
    """Conclusive state exclusion on complex set of pure states."""
    mat_1 = np.array(
        [
            [0.65925619 + 0.0j, 0.38049979 + 0.01670518j, 0.26366168 - 0.1003037j],
            [0.38049979 - 0.01670518j, 0.22003457 + 0.0j, 0.14963473 - 0.06457285j],
            [0.26366168 + 0.1003037j, 0.14963473 + 0.06457285j, 0.12070924 + 0.0j],
        ]
    )

    mat_2 = np.array(
        [
            [0.36978742 + 0.0j, 0.2010124 - 0.07736818j, 0.42433574 - 0.08119137j],
            [0.2010124 + 0.07736818j, 0.12545538 + 0.0j, 0.24765141 + 0.04464623j],
            [0.42433574 + 0.08119137j, 0.24765141 - 0.04464623j, 0.5047572 + 0.0j],
        ]
    )

    mat_3 = np.array(
        [
            [0.34592805 + 0.0j, 0.28877002 + 0.03437698j, 0.31590575 + 0.20468388j],
            [0.28877002 - 0.03437698j, 0.24447252 + 0.0j, 0.28404902 + 0.13947028j],
            [0.31590575 - 0.20468388j, 0.28404902 - 0.13947028j, 0.40959944 + 0.0j],
        ]
    )
    states = [mat_1, mat_2, mat_3]
    probs = [1 / 3, 1 / 3, 1 / 3]

    res = state_exclusion(states, probs, "conclusive")
    assert res > 0


def test_unambiguous_state_exclusion_one_state():
    """Unambiguous state exclusion for single state."""
    mat = bell(0) * bell(0).conj().T
    states = [mat]

    res = state_exclusion(states, probs=None, method="unambiguous")
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_unambiguous_state_exclusion_one_state_vec():
    """Unambiguous state exclusion for single vector state."""
    vec = bell(0)
    states = [vec]

    res = state_exclusion(states, probs=None, method="unambiguous")
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_unambiguous_state_exclusion_three_state():
    """Unambiguous state exclusion for three Bell state density matrices."""
    mat1 = bell(0) * bell(0).conj().T
    mat2 = bell(1) * bell(1).conj().T
    mat3 = bell(2) * bell(2).conj().T
    states = [mat1, mat2, mat3]
    probs = [1 / 3, 1 / 3, 1 / 3]

    res = state_exclusion(states, probs, method="unambiguous")
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_unambiguous_state_exclusion_three_state_vec():
    """Unambiguous state exclusion for three Bell state vectors."""
    mat1 = bell(0)
    mat2 = bell(1)
    mat3 = bell(2)
    states = [mat1, mat2, mat3]
    probs = [1 / 3, 1 / 3, 1 / 3]

    res = state_exclusion(states, probs, method="unambiguous")
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_unambiguous_state_exclusion_complex_three_state_vec():
    """Unambiguous state exclusion on complex set of pure states."""
    mat_1 = np.array(
        [
            [0.37166502 + 0.0j, 0.06990262 + 0.00381928j, 0.44548935 - 0.17369055j],
            [0.06990262 - 0.00381928j, 0.01318651 + 0.0j, 0.0820026 - 0.03724556j],
            [0.44548935 + 0.17369055j, 0.0820026 + 0.03724556j, 0.61514848 + 0.0j],
        ]
    )

    mat_2 = np.array(
        [
            [0.03351844 + 0.0j, 0.08195346 + 0.02701392j, 0.15185457 + 0.0434629j],
            [0.08195346 - 0.02701392j, 0.22215 + 0.0j, 0.40631695 - 0.01611805j],
            [0.15185457 - 0.0434629j, 0.40631695 + 0.01611805j, 0.74433156 + 0.0j],
        ]
    )

    mat_3 = np.array(
        [
            [0.51449115 + 0.0j, 0.23369567 + 0.15751255j, 0.41173315 - 0.02901644j],
            [0.23369567 - 0.15751255j, 0.15437363 + 0.0j, 0.17813679 - 0.13923301j],
            [0.41173315 + 0.02901644j, 0.17813679 + 0.13923301j, 0.33113522 + 0.0j],
        ]
    )
    states = [mat_1, mat_2, mat_3]
    probs = [1 / 3, 1 / 3, 1 / 3]

    res = state_exclusion(states, probs, method="unambiguous")
    np.testing.assert_equal(np.isclose(res, 0, atol=1e-06), True)


def test_state_exclusion_invalid_method():
    """Invalid exclusion method for state exclusion."""
    with np.testing.assert_raises(ValueError):
        mat1 = bell(0)
        mat2 = bell(1)
        mat3 = bell(2)
        states = [mat1, mat2, mat3]
        probs = [1 / 3, 1 / 3, 1 / 3]

        state_exclusion(states, probs, method="not_valid")


if __name__ == "__main__":
    np.testing.run_module_suite()
