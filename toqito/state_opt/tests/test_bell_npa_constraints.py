"""Tests for the bell_npa_constraints function and related helpers."""

import sys

import cvxpy
import pytest
from pytest_mock import MockerFixture

from toqito.state_opt.bell_npa_constraints import _word_to_p_cg_index, bell_npa_constraints
from toqito.state_opt.npa_hierarchy import Symbol, _gen_words


@pytest.mark.parametrize(
    "k, desc, expected_gamma_shape",
    [
        (1, [2, 2, 2, 2], (5, 5)),
        ("1+ab", [2, 2, 2, 2], (9, 9)),
        (2, [2, 2, 2, 2], (13, 13)),
        ("1+aab", [2, 2, 2, 2], (13, 13)),
        (1, [3, 3, 2, 2], (9, 9)),
        ("1+a", [3, 3, 2, 2], (9, 9)),
        ("1+ab", [3, 3, 2, 2], (25, 25)),
    ],
)
def test_bell_npa_constraints_output_structure(k, desc, expected_gamma_shape):
    """Test the output structure and Gamma matrix shape for various NPA levels and scenarios."""
    oa, ob, ma, mb = desc
    p_var_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
    p_var = cvxpy.Variable(p_var_dim, name="p_test")
    constraints = bell_npa_constraints(p_var, desc, k=k)

    assert isinstance(constraints, list)
    assert len(constraints) > 0

    gamma_var = None
    psd_constraints_count = 0
    psd_constraint = None
    for constr in constraints:
        if isinstance(constr, cvxpy.constraints.PSD):
            psd_constraints_count += 1
            psd_constraint = constr

    assert psd_constraints_count == 1, "Expected exactly one PSD constraint for Gamma."
    assert psd_constraint is not None, "Gamma PSD constraint not found."

    psd_arg_vars = psd_constraint.args[0].variables()
    assert len(psd_arg_vars) == 1, "PSD constraint argument should involve only one variable (Gamma)."
    gamma_var = psd_arg_vars[0]

    assert isinstance(gamma_var, cvxpy.Variable), "Extracted Gamma object is not a CVXPY Variable."
    assert gamma_var.name().startswith("Gamma"), "Extracted variable name does not start with 'Gamma'."
    assert gamma_var.shape == expected_gamma_shape

    words = _gen_words(k, oa, ma, ob, mb)
    assert len(words) == expected_gamma_shape[0]


@pytest.mark.parametrize(
    "word, desc, expected_index",
    [
        ((), [2, 2, 2, 2], 0),
        ((Symbol("Alice", 0, 0),), [2, 2, 2, 2], 1),
        ((Symbol("Alice", 1, 0),), [2, 2, 2, 2], 2),
        ((Symbol("Bob", 0, 0),), [2, 2, 2, 2], 3),
        ((Symbol("Bob", 1, 0),), [2, 2, 2, 2], 6),
        ((Symbol("Alice", 0, 0), Symbol("Bob", 0, 0)), [2, 2, 2, 2], 4),
        ((Symbol("Alice", 1, 0), Symbol("Bob", 1, 0)), [2, 2, 2, 2], 8),
        ((Symbol("Alice", 0, 0), Symbol("Alice", 1, 0)), [2, 2, 2, 2], None),
        ((Symbol("Alice", 0, 0), Symbol("Bob", 0, 0), Symbol("Bob", 1, 0)), [2, 2, 2, 2], None),
        ((), [3, 3, 2, 2], 0),
        ((Symbol("Alice", 0, 0),), [3, 3, 2, 2], 1),
        ((Symbol("Alice", 0, 1),), [3, 3, 2, 2], 2),
        ((Symbol("Alice", 1, 0),), [3, 3, 2, 2], 3),
        ((Symbol("Bob", 0, 0),), [3, 3, 2, 2], 5),
        ((Symbol("Bob", 1, 1),), [3, 3, 2, 2], 20),
        ((Symbol("Alice", 0, 0), Symbol("Bob", 0, 0)), [3, 3, 2, 2], 6),
        ((Symbol("Alice", 1, 1), Symbol("Bob", 1, 1)), [3, 3, 2, 2], 24),
    ],
)
def test_word_to_p_cg_index(word, desc, expected_index):
    """Test the mapping from operator words to flattened CG probability vector indices."""
    oa, ob, ma, mb = desc
    assert _word_to_p_cg_index(word, oa, ob, ma, mb) == expected_index


def test_word_to_p_cg_index_bob_explicit():
    """Test the mapping specifically for a Bob-only word and invalid words."""
    desc = [3, 4, 1, 2]
    oa, ob, ma, mb = desc
    word = (Symbol("Bob", 1, 2),)
    expected_index = 18
    assert _word_to_p_cg_index(word, oa, ob, ma, mb) == expected_index

    word_other = (Symbol("Charlie", 1, 2),)
    assert _word_to_p_cg_index(word_other, oa, ob, ma, mb) is None

    word_bob_alice = (Symbol("Bob", 0, 0), Symbol("Alice", 0, 0))
    assert _word_to_p_cg_index(word_bob_alice, oa, ob, ma, mb) is None


def test_bell_npa_constraints_identity_constraint():
    """Test that the constraint Gamma[0, 0] == p_var[0, 0] is correctly generated."""
    desc = [2, 2, 2, 2]
    oa, ob, ma, mb = desc
    p_var_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
    p_var = cvxpy.Variable(p_var_dim, name="p_test_identity")
    constraints = bell_npa_constraints(p_var, desc, k=1)

    assert len(constraints) >= 2

    psd_constraint = constraints[0]
    assert isinstance(psd_constraint, cvxpy.constraints.PSD)
    psd_arg_vars = psd_constraint.args[0].variables()
    assert len(psd_arg_vars) == 1
    gamma_var = psd_arg_vars[0]
    assert isinstance(gamma_var, cvxpy.Variable)
    assert gamma_var.name().startswith("Gamma")

    actual_identity_constraint = constraints[1]
    assert isinstance(actual_identity_constraint, cvxpy.constraints.Equality)

    match_found = False
    try:
        if len(actual_identity_constraint.args) == 2:
            actual_arg0, actual_arg1 = actual_identity_constraint.args
            expected_gamma_arg = gamma_var[0, 0]
            expected_p_arg = p_var[0, 0]

            arg0_str = str(actual_arg0)
            arg1_str = str(actual_arg1)
            expected_gamma_str = str(expected_gamma_arg)
            expected_p_str = str(expected_p_arg)

            if (arg0_str == expected_gamma_str and arg1_str == expected_p_str) or (
                arg0_str == expected_p_str and arg1_str == expected_gamma_str
            ):
                match_found = True
    except Exception:
        pass

    if not match_found:
        actual_constr_str = str(actual_identity_constraint)
        expected_constr_str1 = str(gamma_var[0, 0] == p_var[0, 0])
        expected_constr_str2 = str(p_var[0, 0] == gamma_var[0, 0])
        if actual_constr_str in {expected_constr_str1, expected_constr_str2}:
            match_found = True

    assert match_found, (
        f"Constraint Gamma[0, 0] == p_var[0, 0] structure not found as constraints[1].\n"
        f"Actual constraint: {actual_identity_constraint}"
    )


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 10), reason="unittest.mock.patch string resolution issue in Python 3.10"
)
def test_bell_npa_constraints_value_error(mocker: MockerFixture):
    """Test that a ValueError is raised if the identity word mapping fails internally."""
    desc = [2, 2, 2, 2]
    oa, ob, ma, mb = desc
    p_var_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
    p_var = cvxpy.Variable(p_var_dim, name="p_test_error")

    mocker.patch("toqito.state_opt.bell_npa_constraints._word_to_p_cg_index", return_value=1)
    with pytest.raises(ValueError, match="Internal error: Identity word mapping failed."):
        bell_npa_constraints(p_var, desc, k=1)


@pytest.mark.parametrize(
    "k, desc, expected_gamma_shape_str",
    [
        (1, [2, 2, 2, 2], "(5, 5)"),
        ("1+ab", [2, 2, 2, 2], "(9, 9)"),
        (1, [3, 3, 2, 2], "(9, 9)"),
        ("1+a", [3, 3, 2, 2], "(9, 9)"),
    ],
)
def test_bell_npa_constraints_examples(k, desc, expected_gamma_shape_str):
    """Test the examples provided in the function's docstring."""
    oa, ob, ma, mb = desc
    p_var_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
    p_var = cvxpy.Variable(p_var_dim, name="p_example")
    constraints = bell_npa_constraints(p_var, desc, k=k)

    assert len(constraints) > 0
    psd_constraint = [c for c in constraints if isinstance(c, cvxpy.constraints.PSD)][0]
    gamma_var = psd_constraint.args[0].variables()[0]
    assert str(gamma_var.shape) == expected_gamma_shape_str
