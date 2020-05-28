"""Testing properties of quantum states."""
import numpy as np

from toqito.state_props import is_ensemble
from toqito.state_props import is_mixed
from toqito.state_props import is_mub
from toqito.state_props import is_ppt
from toqito.state_props import is_product_vector
from toqito.state_props import is_pure
from toqito.state_props import concurrence
from toqito.state_props import negativity
from toqito.state_props import schmidt_rank

from toqito.states import basis
from toqito.states import bell
from toqito.states import max_entangled


def test_is_ensemble_true():
    """Test if valid ensemble returns True."""
    rho_0 = np.array([[0.5, 0], [0, 0]])
    rho_1 = np.array([[0, 0], [0, 0.5]])
    states = [rho_0, rho_1]
    np.testing.assert_equal(is_ensemble(states), True)


def test_is_non_ensemble_non_psd():
    """Test if non-valid ensemble returns False."""
    rho_0 = np.array([[0.5, 0], [0, 0]])
    rho_1 = np.array([[-1, -1], [-1, -1]])
    states = [rho_0, rho_1]
    np.testing.assert_equal(is_ensemble(states), False)


def test_is_mixed():
    """Return True for mixed quantum state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    np.testing.assert_equal(is_mixed(rho), True)


def test_is_mub_dim_2():
    """Return True for MUB of dimension 2."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    mub_1 = [e_0, e_1]
    mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), 1 / np.sqrt(2) * (e_0 - e_1)]
    mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), 1 / np.sqrt(2) * (e_0 - 1j * e_1)]
    mubs = [mub_1, mub_2, mub_3]
    np.testing.assert_equal(is_mub(mubs), True)


def test_is_not_mub_dim_2():
    """Return False for non-MUB of dimension 2."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    mub_1 = [e_0, e_1]
    mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), e_1]
    mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), e_0]
    mubs = [mub_1, mub_2, mub_3]
    np.testing.assert_equal(is_mub(mubs), False)


def test_is_mub_invalid_input_len():
    """Tests for invalid input len."""
    with np.testing.assert_raises(ValueError):
        vec_list = [np.array([1, 0])]
        is_mub(vec_list)


def test_is_ppt():
    """Check that PPT matrix returns True."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat), True)


def test_is_ppt_sys():
    """Check that PPT matrix returns True with sys specified."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat, 2), True)


def test_is_ppt_dim_sys():
    """Check that PPT matrix returns True with dim and sys specified."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat, 2, np.round(np.sqrt(mat.size))), True)


def test_is_ppt_tol():
    """Check that PPT matrix returns True."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat, 2, np.round(np.sqrt(mat.size)), 1e-10), True)


def test_entangled_state():
    """Entangled state of dimension 2 will violate PPT criterion."""
    rho = bell(2) * bell(2).conj().T
    np.testing.assert_equal(is_ppt(rho), False)


def test_is_product_entangled_state():
    """Check that is_product_vector returns False for an entangled state."""
    ent_vec = max_entangled(3)
    np.testing.assert_equal(is_product_vector(ent_vec), False)


def test_is_product_entangled_state_2_sys():
    """Check that dimension argument as list is supported."""
    ent_vec = max_entangled(4)
    np.testing.assert_equal(is_product_vector(ent_vec, dim=[4, 4]), False)


def test_is_product_entangled_state_3_sys():
    """Check that dimension argument as list is supported."""
    ent_vec = max_entangled(4)
    np.testing.assert_equal(is_product_vector(ent_vec, dim=[2, 2, 2, 2]), False)


def test_is_product_separable_state():
    """Check that is_product_vector returns True for a separable state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    sep_vec = (
        1
        / 2
        * (
            np.kron(e_0, e_0)
            - np.kron(e_0, e_1)
            - np.kron(e_1, e_0)
            + np.kron(e_1, e_1)
        )
    )
    np.testing.assert_equal(is_product_vector(sep_vec), True)


def test_is_pure_state():
    """Ensure that pure Bell state returns True."""
    rho = bell(0) * bell(0).conj().T
    np.testing.assert_equal(is_pure(rho), True)


def test_is_pure_list():
    """Check that list of pure states returns True."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)

    e0_dm = e_0 * e_0.conj().T
    e1_dm = e_1 * e_1.conj().T
    e2_dm = e_2 * e_2.conj().T

    np.testing.assert_equal(is_pure([e0_dm, e1_dm, e2_dm]), True)


def test_is_pure_not_pure_state():
    """Check that non-pure state returns False."""
    rho = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_pure(rho), False)


def test_is_pure_not_pure_list():
    """Check that list of non-pure states return False."""
    rho = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sigma = np.array([[1, 2, 3], [10, 11, 12], [7, 8, 9]])
    np.testing.assert_equal(is_pure([rho, sigma]), False)


def test_concurrence_entangled():
    """The concurrence on maximally entangled Bell state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec * u_vec.conj().T

    res = concurrence(rho)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_concurrence_separable():
    """The concurrence of a product state is zero."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    v_vec = np.kron(e_0, e_1)
    sigma = v_vec * v_vec.conj().T

    res = concurrence(sigma)
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_concurrence_invalid_dim():
    """Tests for invalid dimension inputs."""
    with np.testing.assert_raises(ValueError):
        rho = np.identity(5)
        concurrence(rho)


def test_negativity_rho():
    """Test for negativity on rho."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    np.testing.assert_equal(np.isclose(negativity(test_input_mat), 1 / 2), True)


def test_negativity_rho_dim_int():
    """Test for negativity on rho."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    np.testing.assert_equal(np.isclose(negativity(test_input_mat, 2), 1 / 2), True)


def test_negativity_invalid_rho_dim_int():
    """Invalid dim parameters."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        negativity(test_input_mat, 5)


def test_negativity_invalid_rho_dim_vec():
    """Invalid dim parameters."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        negativity(test_input_mat, [2, 5])


def test_schmidt_rank_bell_state():
    """
    Computing the Schmidt rank of the entangled Bell state should yield a
    value greater than 1.
    """
    rho = bell(0).conj().T * bell(0)
    np.testing.assert_equal(schmidt_rank(rho) > 1, True)


def test_schmidt_rank_singlet_state():
    """
    Computing the Schmidt rank of the entangled singlet state should yield
    a value greater than 1.
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))
    rho = rho.conj().T * rho
    np.testing.assert_equal(schmidt_rank(rho) > 1, True)


def test_schmidt_rank_separable_state():
    """
    Computing the Schmidt rank of a separable state should yield a value
    equal to 1.
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_01 = np.kron(e_0, e_1)
    e_10 = np.kron(e_1, e_0)
    e_11 = np.kron(e_1, e_1)
    rho = 1 / 2 * (e_00 - e_01 - e_10 + e_11)
    rho = rho.conj().T * rho
    np.testing.assert_equal(schmidt_rank(rho) == 1, True)


if __name__ == "__main__":
    np.testing.run_module_suite()
