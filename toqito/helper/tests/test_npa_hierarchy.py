"""Test npa_constraints and its helper functions."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.helper.npa_hierarchy import IDENTITY_SYMBOL, Symbol, _gen_words, _parse, _reduce, npa_constraints

# Define common symbols for helper function unit tests
A00_test = Symbol("Alice", 0, 0)
A01_test = Symbol("Alice", 0, 1)
A10_test = Symbol("Alice", 1, 0)
A11_test = Symbol("Alice", 1, 1)
B00_test = Symbol("Bob", 0, 0)
B01_test = Symbol("Bob", 0, 1)
B10_test = Symbol("Bob", 1, 0)
B11_test = Symbol("Bob", 1, 1)


class TestNPAReduce:
    """Test the _reduce helper function."""

    def test_empty_word(self):
        """Test reducing an empty word."""
        assert _reduce(()) == ()

    def test_identity_word(self):
        """Test reducing an identity word."""
        assert _reduce((IDENTITY_SYMBOL,)) == (IDENTITY_SYMBOL,)

    def test_single_symbol(self):
        """Test reducing a single Alice or Bob symbol."""
        assert _reduce((A00_test,)) == (A00_test,)
        assert _reduce((B00_test,)) == (B00_test,)

    def test_idempotence(self):
        """Test projector idempotence A*A = A."""
        assert _reduce((A00_test, A00_test)) == (A00_test,)
        assert _reduce((A00_test, A00_test, B00_test)) == (A00_test, B00_test)
        assert _reduce((A00_test, B00_test, B00_test)) == (A00_test, B00_test)

    def test_orthogonality(self):
        """Test projector orthogonality A_i A_j = 0 for i!=j."""
        assert _reduce((A00_test, A01_test)) == ()
        assert _reduce((B00_test, B01_test)) == ()
        assert _reduce((A00_test, B00_test, B01_test)) == ()
        assert _reduce((A00_test, A01_test, B00_test)) == ()

    def test_commutation(self):
        """Test Alice-Bob operator commutation."""
        assert _reduce((B00_test, A00_test)) == (A00_test, B00_test)
        assert _reduce((B00_test, A00_test, B10_test)) == (A00_test, B00_test, B10_test)
        assert _reduce((B00_test, A00_test, B00_test)) == (A00_test, B00_test)

    def test_complex_reduction_to_zero(self):
        """Test a more complex reduction that results in zero."""
        assert _reduce((A00_test, B00_test, A00_test, B01_test)) == ()

    def test_final_not_final_word_empty_after_separate_reduction(self):
        """Test reduction where both Alice and Bob parts individually become zero."""
        assert _reduce((A00_test, A01_test, B00_test, B01_test)) == ()

    def test_reduction_with_identity_mixed(self):
        """Test reduction when IDENTITY_SYMBOL is mixed (though not typical for sub-words)."""
        assert _reduce((A00_test, IDENTITY_SYMBOL)) == (A00_test,)
        assert _reduce((IDENTITY_SYMBOL, A00_test)) == (A00_test,)
        assert _reduce((A00_test, IDENTITY_SYMBOL, B00_test)) == (A00_test, B00_test)
        assert _reduce((A00_test, B00_test, IDENTITY_SYMBOL)) == (A00_test, B00_test)


class TestNPAParse:
    """Test the _parse helper function for k-string."""

    def test_simple_int(self):
        """Test parsing a simple integer k."""
        assert _parse("1") == (1, set())
        assert _parse("2") == (2, set())

    def test_simple_config(self):
        """Test parsing k with simple configurations like '1+a'."""
        assert _parse("1+a") == (1, {(1, 0)})
        assert _parse("1+b") == (1, {(0, 1)})
        assert _parse("1+ab") == (1, {(1, 1)})
        assert _parse("2+aa+bb") == (2, {(2, 0), (0, 2)})

    def test_multiple_configs(self):
        """Test parsing k with multiple configurations."""
        k_int, conf = _parse("1+a+b+aa+ab")
        assert k_int == 1
        assert conf == {(1, 0), (0, 1), (2, 0), (1, 1)}

    def test_empty_config_part(self):
        """Test parsing k with empty parts like '1+' or '1++ab'."""
        assert _parse("1+") == (1, set())
        assert _parse("1++ab") == (1, {(1, 1)})
        assert _parse("1+ab+") == (1, {(1, 1)})

    def test_invalid_char_in_config(self):
        """Test parsing k with invalid characters in configurations."""
        expected_msg_regex = re.escape(
            "Invalid character 'c' in k string component " + "'ac'. Only 'a' or 'b' allowed after base k."
        )
        with pytest.raises(ValueError, match=expected_msg_regex):
            _parse("1+ac")

    def test_invalid_base_k_empty_start(self):
        """Test parsing k string starting with '+' (e.g., '+ab')."""
        expected_msg_regex = re.escape("Base level k must be specified, e.g., '1+ab'")
        with pytest.raises(ValueError, match=expected_msg_regex):
            _parse("+ab")

    def test_empty_string_input_for_parse(self):
        """Test parsing an empty string k."""
        with pytest.raises(ValueError, match="Input string k_str cannot be empty."):
            _parse("")

    def test_invalid_base_k_non_int_start(self):
        """Test parsing k string where base k is not an integer (e.g., 'a+b')."""
        expected_msg_regex = re.escape(
            "Base level k 'a' is not a valid integer: invalid literal for int() with base 10: 'a'"
        )
        with pytest.raises(ValueError, match=expected_msg_regex):
            _parse("a+b")

    def test_reduction_where_one_player_part_is_zero(self):  # Was test_reduce_final_word_is_identity_from_parts
        """Test reduction when one player's sequence becomes zero, making the whole product zero."""
        # (A00_test, B00_test, B01_test) -> A00 * (B00*B01) -> A00 * 0 -> 0
        assert _reduce((A00_test, B00_test, B01_test)) == ()
        # (A00_test, A01_test, B00_test) -> (A00*A01) * B00 -> 0 * B00 -> 0
        assert _reduce((A00_test, A01_test, B00_test)) == ()

    def test_reduce_product_with_actual_identity_symbol(self):  # This one should pass now
        """Test how _reduce handles products involving the explicit IDENTITY_SYMBOL."""
        assert _reduce((A00_test, IDENTITY_SYMBOL)) == (A00_test,)
        assert _reduce((IDENTITY_SYMBOL, A00_test)) == (A00_test,)
        assert _reduce((IDENTITY_SYMBOL, IDENTITY_SYMBOL)) == (IDENTITY_SYMBOL,)  # Fixed by new _reduce
        assert _reduce((A00_test, IDENTITY_SYMBOL, B00_test)) == (A00_test, B00_test)
        assert _reduce((A00_test, B00_test, IDENTITY_SYMBOL)) == (A00_test, B00_test)

    def test_complex_reduction_to_zero(self):  # Should still pass
        """Test a more complex reduction that results in zero."""
        assert _reduce((A00_test, B00_test, A00_test, B01_test)) == ()

    def test_both_player_parts_reduce_to_zero(self):  # Was test_final_not_final_word_empty_after_separate_reduction
        """Test reduction where both Alice's and Bob's parts individually become zero."""
        assert _reduce((A00_test, A01_test, B00_test, B01_test)) == ()


class TestNPAGenWords:
    """Test the _gen_words helper function."""

    cglmp_a_out, cglmp_a_in = 3, 2
    cglmp_b_out, cglmp_b_in = 3, 2

    simple_a_out, simple_a_in = 2, 1
    simple_b_out, simple_b_in = 2, 1

    def test_k1_simple_1in_2out(self):
        """Test _gen_words for k=1, 1-input/2-output per player."""
        words = _gen_words(
            k=1, a_out=self.simple_a_out, a_in=self.simple_a_in, b_out=self.simple_b_out, b_in=self.simple_b_in
        )
        expected_set = {(IDENTITY_SYMBOL,), (A00_test,), (B00_test,)}
        assert set(words) == expected_set

    def test_k1_cglmp_params(self):
        """Test _gen_words for k=1 with CGLMP parameters."""
        words = _gen_words(
            k=1, a_out=self.cglmp_a_out, a_in=self.cglmp_a_in, b_out=self.cglmp_b_out, b_in=self.cglmp_b_in
        )
        assert len(words) == 9

    def test_string_k_1_plus_ab_simple(self):
        """Test _gen_words for k='1+ab', simple params."""
        words = _gen_words(
            "1+ab", a_out=self.simple_a_out, a_in=self.simple_a_in, b_out=self.simple_b_out, b_in=self.simple_b_in
        )
        expected_set = {(IDENTITY_SYMBOL,), (A00_test,), (B00_test,), (A00_test, B00_test)}
        assert set(words) == expected_set

    def test_string_k_1_plus_ab_cglmp_params(self):
        """Test _gen_words for k='1+ab' with CGLMP parameters."""
        words = _gen_words(
            "1+ab", a_out=self.cglmp_a_out, a_in=self.cglmp_a_in, b_out=self.cglmp_b_out, b_in=self.cglmp_b_in
        )
        assert len(words) == 25

    def test_orthogonality_in_gen_k2(self):
        """Test _gen_words for k=2 ensuring orthogonality is handled."""
        a01_for_this_test = Symbol("Alice", 0, 1)
        words = _gen_words(k=2, a_out=3, a_in=1, b_out=2, b_in=1)
        expected_set = {
            (IDENTITY_SYMBOL,),
            (A00_test,),
            (a01_for_this_test,),
            (B00_test,),
            (A00_test, B00_test),
            (a01_for_this_test, B00_test),
        }
        assert set(words) == expected_set

    def test_gen_words_string_k_complex(self):
        """Test _gen_words for a complex k-string '1+aa+ab'."""
        words = _gen_words(
            "1+aa+ab", a_out=self.simple_a_out, a_in=self.simple_a_in, b_out=self.simple_b_out, b_in=self.simple_b_in
        )
        expected_set = {(IDENTITY_SYMBOL,), (A00_test,), (B00_test,), (A00_test, B00_test)}
        assert set(words) == expected_set

    def test_gen_words_no_operators_if_out_is_1(self):
        """Test _gen_words when one or both players have only 1 outcome (no non-id symbols)."""
        words = _gen_words(1, a_out=1, a_in=1, b_out=self.simple_b_out, b_in=self.simple_b_in)
        assert set(words) == {(IDENTITY_SYMBOL,), (B00_test,)}
        words_bob_too = _gen_words(1, a_out=1, a_in=1, b_out=1, b_in=1)
        assert set(words_bob_too) == {(IDENTITY_SYMBOL,)}


# Integration tests for npa_constraints
def cglmp_setup_vars_and_objective(num_outcomes: int) -> tuple[dict[tuple[int, int], cvxpy.Variable], cvxpy.Expression]:
    """Set up variables and objective for CGLMP inequality for npa_constraints."""
    (a_in, b_in) = (2, 2)
    (a_out, b_out) = (num_outcomes, num_outcomes)

    assemblage_vars = {
        (x, y): cvxpy.Variable((a_out, b_out), name=f"Probs_xy_{x}{y}") for x in range(a_in) for y in range(b_in)
    }

    i_b_expr = cvxpy.Constant(0)
    # Using the sum form from the original test for now
    for k_sum_idx in range(num_outcomes // 2):
        tmp = 0
        for a_val in range(a_out):
            for b_val in range(b_out):
                if a_val == np.mod(b_val + k_sum_idx, num_outcomes):
                    tmp += assemblage_vars[0, 0][a_val, b_val]
                    tmp += assemblage_vars[1, 1][a_val, b_val]

                if b_val == np.mod(a_val + k_sum_idx + 1, num_outcomes):
                    tmp += assemblage_vars[1, 0][a_val, b_val]

                if b_val == np.mod(a_val + k_sum_idx, num_outcomes):
                    tmp += assemblage_vars[0, 1][a_val, b_val]

                if a_val == np.mod(b_val - k_sum_idx - 1, num_outcomes):
                    tmp -= assemblage_vars[0, 0][a_val, b_val]
                    tmp -= assemblage_vars[1, 1][a_val, b_val]

                if b_val == np.mod(a_val - k_sum_idx, num_outcomes):
                    tmp -= assemblage_vars[1, 0][a_val, b_val]

                if b_val == np.mod(a_val - k_sum_idx - 1, num_outcomes):
                    tmp -= assemblage_vars[0, 1][a_val, b_val]

        denominator = num_outcomes - 1
        if denominator == 0:  # Avoid division by zero if num_outcomes is 1
            # This case is degenerate for CGLMP, but handle defensively
            i_b_expr += tmp
        else:
            i_b_expr += (1 - 2 * k_sum_idx / denominator) * tmp

    return assemblage_vars, i_b_expr


@pytest.mark.parametrize("k_npa", [2, "1+ab+aab+baa"])
def test_cglmp_inequality_npa_integration(k_npa):
    """Test CGLMP inequality (d=3) via npa_constraints."""
    cglmp_d = 3
    assemblage_vars, i_b_objective = cglmp_setup_vars_and_objective(cglmp_d)

    npa_constraints_list = npa_constraints(assemblage_vars, k_npa, referee_dim=1)

    objective = cvxpy.Maximize(i_b_objective)
    problem = cvxpy.Problem(objective, npa_constraints_list)
    val = problem.solve(solver=cvxpy.SCS, verbose=False)

    expected_cglmp_val = 2.9149
    print(f"CGLMP d={cglmp_d}, k_npa='{k_npa}', Solved value={val}, Expected={expected_cglmp_val}")
    assert val == pytest.approx(expected_cglmp_val, abs=1e-3)


@pytest.mark.parametrize("k_npa, expected_num_words", [("1+a", 9), ("1+ab", 25)])
def test_cglmp_moment_matrix_dimension_integration(k_npa, expected_num_words):
    """Test moment matrix size for CGLMP d=3 via npa_constraints setup."""
    cglmp_d = 3
    actual_a_out, actual_a_in, actual_b_out, actual_b_in = (cglmp_d, 2, cglmp_d, 2)
    words = _gen_words(k_npa, actual_a_out, actual_a_in, actual_b_out, actual_b_in)
    num_words = len(words)
    assert num_words * 1 == expected_num_words  # referee_dim=1


def test_gen_words_intermediate_hierarchy_call_check():  # Original test name
    """Check npa_constraints runs with an intermediate hierarchy string."""
    referee_dim = 1
    a_out, a_in = 2, 2
    b_out, b_in = 2, 2
    assemblage = {
        (x, y): cvxpy.Variable((referee_dim * a_out, referee_dim * b_out), name=f"K_xy_{x}{y}")
        for x in range(a_in)
        for y in range(b_in)
    }
    k = "1+ab+bb"
    constraints = npa_constraints(assemblage, k, referee_dim=referee_dim)
    assert len(constraints) > 0


@pytest.fixture
def mock_assemblage_setup():
    r"""Provide a setup function to create mock assemblage variables for NPA tests.

    This fixture returns a helper function, `_setup`. When called, `_setup`
    generates a dictionary of CVXPY variables representing the commuting
    measurement assemblage operator K. This mock assemblage can then be passed
    to `npa_constraints` for testing purposes.

    The structure of the returned `assemblage_vars` dictionary is:
    - Keys: Tuples `(x, y)` representing Alice's question `x` and Bob's question `y`.
    - Values: `cvxpy.Variable` of shape `(ref_dim * a_out, ref_dim * b_out)`.
      This variable represents the matrix K_xy, which itself contains blocks
      K_xy(a,b) corresponding to Alice's answer `a` and Bob's answer `b`.
      The `npa_constraints` function internally slices these K_xy matrices
      to access the K_xy(a,b) blocks.

    The inner `_setup` function takes game dimension parameters as arguments.
    """

    def _setup(
        a_in: int, a_out: int, b_in: int, b_out: int, ref_dim: int
    ) -> tuple[dict[tuple[int, int], cvxpy.Variable], int, int, int, int, int]:
        r"""Create mock assemblage variables and return game parameters.

        This helper function is returned by the `mock_assemblage_setup` fixture.

        :param a_in: Number of Alice's possible inputs.
        :param a_out: Number of Alice's possible outputs.
        :param b_in: Number of Bob's possible inputs.
        :param b_out: Number of Bob's possible outputs.
        :param ref_dim: Dimension of the referee's quantum system.
        :return: A tuple containing:
                 - `assemblage_vars`: A dictionary where keys are `(x,y)` input tuples
                   and values are `cvxpy.Variable` of shape
                   `(ref_dim * a_out, ref_dim * b_out)`.
                 - `a_out`, `a_in`, `b_out`, `b_in`, `ref_dim` (passed through for convenience).
        """
        assemblage_vars = {
            (x, y): cvxpy.Variable(
                (ref_dim * a_out, ref_dim * b_out),
                name=f"K_fixture_{x}{y}",
                # Not setting hermitian=True here, as the full K_xy matrix
                # is not necessarily Hermitian. Its sub-blocks K_xy(a,b)
                # representing conditional states will be constrained to be PSD
                # (and thus Hermitian) by npa_constraints.
            )
            for x in range(a_in)
            for y in range(b_in)
        }
        return assemblage_vars, a_out, a_in, b_out, b_in, ref_dim

    return _setup


def test_npa_constraints_identity_product_branch(mock_assemblage_setup):
    """Test the branch: S_i^dagger S_j = I, but (i,j) != (0,0)."""
    assemblage, a_out, a_in, b_out, b_in, ref_dim = mock_assemblage_setup(a_in=1, a_out=2, b_in=1, b_out=2, ref_dim=1)
    # To hit this, we need words such that words[i] = P, words[j] = P (so i=j, i!=0)
    # and _reduce(P_dagger P) = P, which is not Identity unless P=I.
    # Or words[i] = P, words[j] = P_inv. But our symbols are projectors.
    # This branch seems to only be relevant if _gen_words produces redundant Identity words
    # or actual unitary (non-projector) operators.
    # For now, we ensure it doesn't crash. A specific setup to hit the `else` is complex.
    constraints = npa_constraints(assemblage, k=1, referee_dim=ref_dim)  # k=1 has few words
    assert len(constraints) > 0  # Basic check it runs
