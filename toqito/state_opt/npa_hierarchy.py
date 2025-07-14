"""Generates the NPA constraints."""

from collections import namedtuple
from itertools import product

import cvxpy

Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])
IDENTITY_SYMBOL = Symbol("", None, None)  # Explicit identity symbol
PLAYERS = ("Alice", "Bob")


def _reduce(word: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    """Reduce an operator word to its canonical form using NPA rules.

    Identity: I*S = S*I = S, I*I = I
    Commutation: Alice operators commute with Bob operators. Canonical form: A...AB...B
    Orthogonality: P_x,a P_x,b = 0 if a != b (for same player x)
    Idempotence: P_x,a P_x,a = P_x,a (for same player x)
    """
    if not word:
        return ()

    # Initial pass to filter out identities IF other ops are present
    current_list = [s for s in word if s != IDENTITY_SYMBOL]
    if not current_list:  # Original word was all identities or empty
        return (IDENTITY_SYMBOL,) if any(s == IDENTITY_SYMBOL for s in word) else ()

    # Canonical player order (Alice then Bob), preserving original relative internal order
    alice_ops = [s for s in current_list if s.player == "Alice"]
    bob_ops = [s for s in current_list if s.player == "Bob"]
    current_list = alice_ops + bob_ops  # This is now a list of Symbol objects

    # Iteratively apply reduction rules until no more changes occur
    while True:
        len_before_pass = len(current_list)
        next_pass_list = []
        idx = 0
        made_change_in_pass = False

        while idx < len(current_list):
            s_x = current_list[idx]

            if idx + 1 < len(current_list):
                s_y = current_list[idx + 1]
                # Only apply if s_x and s_y are from the same player.
                if s_x == s_y and s_x.player in PLAYERS:  # s_x != IDENTITY_SYMBOL
                    next_pass_list.append(s_x)
                    idx += 2  # Consumed s_x, s_y; added s_x
                    made_change_in_pass = True
                    continue
                # Rule 2: Orthogonality (S_x,a S_x,b = 0 if a!=b, for same player and question)
                elif (
                    s_x.player == s_y.player
                    and s_x.player in PLAYERS  # Ensure not identity
                    and s_x.question == s_y.question
                    and s_x.answer != s_y.answer
                ):
                    return ()  # Entire word becomes zero
                else:
                    # No reduction for this pair, keep s_x
                    next_pass_list.append(s_x)
                    idx += 1
            else:
                # Last element, just append it
                next_pass_list.append(s_x)
                idx += 1

        current_list = next_pass_list
        if not made_change_in_pass and len(current_list) == len_before_pass:  # Stable
            break

    return tuple(current_list) if current_list else ()


def _parse(k_str: str) -> tuple[int, set[tuple[int, int]]]:
    if not k_str:  # Explicitly handle empty string input for k_str
        raise ValueError("Input string k_str cannot be empty.")
    parts = k_str.split("+")
    if not parts[0] or parts[0] == "":  # Check if the first part (base_k) is empty
        raise ValueError("Base level k must be specified, e.g., '1+ab'")
    try:
        base_k = int(parts[0])
    except ValueError as e:
        raise ValueError(f"Base level k '{parts[0]}' is not a valid integer: {e}") from e

    conf = set()
    if len(parts) == 1 and base_k >= 0:  # e.g. "0", "1"
        pass  # conf remains empty, which is correct.

    for val_content in parts[1:]:  # Process each part after the base_k
        cnt_a, cnt_b = 0, 0
        if not val_content:  # Handles "1++ab" -> parts like '', skip these
            continue
        # If val_content is an empty string (e.g., from "0+", "1++a"),
        # cnt_a and cnt_b will remain 0, and (0,0) will be added to conf.
        for char_val in val_content:  # Loop over empty string does nothing
            if char_val == "a":
                cnt_a += 1
            elif char_val == "b":
                cnt_b += 1
            else:
                raise ValueError(
                    f"Invalid character '{char_val}' in k string component "
                    + f"'{val_content}'. Only 'a' or 'b' allowed after base k."
                )
        conf.add((cnt_a, cnt_b))
    return base_k, conf


def _gen_words(k: int | str, a_out: int, a_in: int, b_out: int, b_in: int) -> list[tuple[Symbol, ...]]:
    # Symbols for non-identity measurements (last outcome is dependent)
    alice_symbols = [Symbol("Alice", x, a) for x in range(a_in) for a in range(a_out - 1)]
    bob_symbols = [Symbol("Bob", y, b) for y in range(b_in) for b in range(b_out - 1)]

    words = set([(IDENTITY_SYMBOL,)])  # Start with identity operator

    k_int = k
    configurations = set()

    if isinstance(k, str):
        k_int, configurations = _parse(k)

    # Loop 1: Generate words up to length k_int from the hierarchy
    for length in range(0, k_int + 1):  # Lengths 1, ..., k_int
        for alice_len in range(length + 1):
            bob_len = length - alice_len

            # Generate Alice's part
            # If alice_len is 0, product yields one item: ()
            for word_a_tuple in product(alice_symbols, repeat=alice_len):
                reduced_a = _reduce(word_a_tuple)
                # Alice's part (non-empty originally) reduced to zero
                if reduced_a == () and alice_len > 0:
                    continue

                # Generate Bob's part
                # If bob_len is 0, product yields one item: ()
                for word_b_tuple in product(bob_symbols, repeat=bob_len):
                    reduced_b = _reduce(word_b_tuple)
                    # Bob's part (non-empty originally) reduced to zero
                    if reduced_b == () and bob_len > 0:
                        continue

                    if not reduced_a and not reduced_b:  # Both parts are empty (e.g. alice_len=0, bob_len=0)
                        # This means the total length of operators is 0.
                        final_word = (IDENTITY_SYMBOL,)
                    else:
                        # _reduce will put Alice operators before Bob operators if somehow mixed,
                        # and apply rules. It also handles identity filtering if I was part of word.
                        # Here, reduced_a + reduced_b is already A...AB...B (or just A...A or B...B).
                        final_word = _reduce(reduced_a + reduced_b)
                    words.add(final_word)

    # Loop 2: Add words from specific configurations (e.g., "1+ab" means k_int=1, configurations={(1,1)})
    for alice_len_conf, bob_len_conf in configurations:
        if alice_len_conf == 0 and bob_len_conf == 0 and k_int == 0 and (IDENTITY_SYMBOL,) in words:
            pass  # The set `words` will handle duplicates from k_int loop vs config loop.

        for word_a_tuple in product(alice_symbols, repeat=alice_len_conf):
            reduced_a = _reduce(word_a_tuple)
            if reduced_a == () and alice_len_conf > 0:
                continue

            for word_b_tuple in product(bob_symbols, repeat=bob_len_conf):
                reduced_b = _reduce(word_b_tuple)
                if reduced_b == () and bob_len_conf > 0:
                    continue

                # Combine and add as in the main loop
                # Both parts are empty (e.g. alice_len_conf=0, bob_len_conf=0)
                if not reduced_a and not reduced_b:
                    # Should not happen if _parse filters (0,0) from conf
                    final_word = (IDENTITY_SYMBOL,)

                else:
                    final_word = _reduce(reduced_a + reduced_b)

                words.add(final_word)

    # If `words` contains `()`, filter it out before converting to list.
    words = {w for w in words if w != ()}
    # Convert set to list, then sort.
    # Make sure (IDENTITY_SYMBOL,) is always at index 0.
    list_of_words = list(words)
    list_of_words.remove((IDENTITY_SYMBOL,))
    # Sort remaining words: typically by length, then by content.
    # Sorting tuples of Symbols needs a consistent key.
    # repr(s) can give a consistent string for sorting.
    list_of_words.sort(key=lambda w: (len(w), tuple(repr(s) for s in w)))
    return [(IDENTITY_SYMBOL,)] + list_of_words


def _is_zero(word: tuple[Symbol, ...]) -> bool:
    # An empty tuple after reduction means the operator product is zero.
    return len(word) == 0


def _is_identity(word: tuple[Symbol, ...]) -> bool:
    return word == (IDENTITY_SYMBOL,)


def _is_meas(word: tuple[Symbol, ...]) -> bool:
    # Expects a reduced word: (Alice_Symbol, Bob_Symbol)
    if len(word) == 2:
        s_a, s_b = word
        return s_a.player == "Alice" and s_b.player == "Bob"
    return False


def _is_meas_on_one_player(word: tuple[Symbol, ...]) -> bool:
    # Expects a reduced word: (Alice_Symbol,) or (Bob_Symbol,)
    if len(word) == 1:
        s = word[0]
        return s.player in PLAYERS  # Excludes IDENTITY_SYMBOL
    return False


# _get_nonlocal_game_params remains the same as in npa_constraints_fix
def _get_nonlocal_game_params(
    assemblage: dict[tuple[int, int], cvxpy.Variable], referee_dim: int = 1
) -> tuple[int, int, int, int]:
    a_in, b_in = max(assemblage.keys())
    a_in += 1
    b_in += 1
    operator = next(iter(assemblage.values()))
    a_out = operator.shape[0] // referee_dim
    b_out = operator.shape[1] // referee_dim
    return a_out, a_in, b_out, b_in


def npa_constraints(
    assemblage: dict[tuple[int, int], cvxpy.Variable], k: int | str = 1, referee_dim: int = 1, no_signaling: bool = True
) -> list[cvxpy.constraints.constraint.Constraint]:
    r"""Generate the constraints specified by the NPA hierarchy up to a finite level.

    :footcite:`Navascues_2008_AConvergent`

    You can determine the level of the hierarchy by a positive integer or a string
    of a form like "1+ab+aab", which indicates that an intermediate level of the hierarchy
    should be used, where this example uses all products of 1 measurement, all products of
    one Alice and one Bob measurement, and all products of two Alice and one Bob measurement.

    The commuting measurement assemblage operator must be given as a dictionary. The keys are
    tuples of Alice and Bob questions :math:`x, y` and the values are cvxpy Variables which
    are matrices with entries:

    .. math::
        K_{xy}\Big(i + a \cdot dim_R, j + b \cdot dim_R \Big) =
        \langle i| \text{Tr}_{\mathcal{H}} \Big( \big(
            I_R \otimes A_a^x B_b^y \big) \sigma \Big) |j \rangle

    References
    ==========
    .. footbibliography::


    :param assemblage: The commuting measurement assemblage operator.
    :param k: The level of the NPA hierarchy to use (default=1).
    :param referee_dim: The dimension of the referee's quantum system (default=1).
    :return: A list of cvxpy constraints.

    """
    a_out, a_in, b_out, b_in = _get_nonlocal_game_params(assemblage, referee_dim)

    words = _gen_words(k, a_out, a_in, b_out, b_in)
    dim = len(words)

    if dim == 0:
        # Should not happen if IDENTITY_SYMBOL is always included
        raise ValueError("Generated word list is empty. Check _gen_words logic.")

    # Moment matrix (Gamma matrix in :footcite:`Navascues_2008_AConvergent`)
    # moment_matrix_R block corresponds to E[S_i^dagger S_j]
    moment_matrix_R = cvxpy.Variable((referee_dim * dim, referee_dim * dim), hermitian=True, name="R")

    # Referee's effective state rho_R = E[I]
    # This is the (0,0) block of moment_matrix_R since words[0] is Identity
    rho_R_referee = moment_matrix_R[0:referee_dim, 0:referee_dim]

    # Ensure rho_R_referee is a valid quantum state
    constraints = [
        cvxpy.trace(rho_R_referee) == 1,
        rho_R_referee >> 0,
        moment_matrix_R >> 0,
    ]

    # Store relations for (S_i^dagger S_j) -> block_index in moment_matrix_R
    # This helps enforce Γ(S_i^dagger S_j) = Γ(S_k^dagger S_l) if products are algebraically equal
    seen_reduced_products = {}

    for i in range(dim):
        for j in range(i, dim):  # Iterate over upper triangle including diagonal
            word_i_conj = tuple(reversed(words[i]))  # S_i^dagger

            # The product S_i^dagger S_j
            # For _reduce, ensure no IDENTITY_SYMBOL unless it's the only element.
            # If word_i_conj is (ID,), S_i_dagger_S_j is S_j. If word_j is (ID,), it's S_i_dagger.
            # If both are (ID,), product is (ID,).

            product_unreduced = []
            if word_i_conj != (IDENTITY_SYMBOL,):
                product_unreduced.extend(list(word_i_conj))
            if words[j] != (IDENTITY_SYMBOL,):
                product_unreduced.extend(list(words[j]))

            # This happens if both words[i] and words[j] were IDENTITY_SYMBOL
            if not product_unreduced:
                product_S_i_adj_S_j = (IDENTITY_SYMBOL,)
            else:
                product_S_i_adj_S_j = _reduce(tuple(product_unreduced))

            # Moment matrix (Gamma matrix in NPA paper :footcite:`Navascues_2008_AConvergent` - arXiv:0803.4290)
            # This hierarchy can be generalized, e.g., to incorporate referee systems
            # as seen in extended nonlocal games (see, e.g., F. Speelman's thesis, :footcite:`Speelman_2016_Position`).
            current_block = moment_matrix_R[
                i * referee_dim : (i + 1) * referee_dim, j * referee_dim : (j + 1) * referee_dim
            ]

            if _is_zero(product_S_i_adj_S_j):  # Product is algebraically zero
                constraints.append(current_block == 0)
            elif _is_identity(product_S_i_adj_S_j):  # Product is identity operator
                # This occurs for (i,j) where S_i^dagger S_j = I. e.g. S_i = S_j and S_i is unitary (proj).
                # Or i=0, j=0 (I^dagger I = I).
                # This means current_block should be rho_R_referee if product_S_i_adj_S_j is I
                constraints.append(current_block == rho_R_referee)

            # Product is A_a^x B_b^y
            elif _is_meas(product_S_i_adj_S_j):
                alice_symbol, bob_symbol = product_S_i_adj_S_j
                constraints.append(
                    current_block
                    == assemblage[alice_symbol.question, bob_symbol.question][
                        alice_symbol.answer * referee_dim : (alice_symbol.answer + 1) * referee_dim,
                        bob_symbol.answer * referee_dim : (bob_symbol.answer + 1) * referee_dim,
                    ]
                )
            # Product is A_a^x or B_b^y (i.e., only one player involved)
            elif _is_meas_on_one_player(product_S_i_adj_S_j):  # Product is A_a^x or B_b^y
                symbol = product_S_i_adj_S_j[0]
                if symbol.player == "Alice":
                    # Sum over Bob's outcomes for a fixed Bob question (e.g., y=0)
                    # E[A_a^x] = sum_b K_x0(a,b)
                    sum_over_bob_outcomes = sum(
                        assemblage[symbol.question, 0][  # Assuming y=0 for Bob's marginal
                            symbol.answer * referee_dim : (symbol.answer + 1) * referee_dim,
                            b_ans * referee_dim : (b_ans + 1) * referee_dim,
                        ]
                        for b_ans in range(b_out)
                    )
                    constraints.append(current_block == sum_over_bob_outcomes)
                else:  # elif symbol.player == "Bob":
                    # Sum over Alice's outcomes for a fixed Alice question (e.g., x=0)
                    # E[B_b^y] = sum_a K_0y(a,b)
                    sum_over_alice_outcomes = sum(
                        assemblage[0, symbol.question][  # Assuming x=0 for Alice's marginal
                            a_ans * referee_dim : (a_ans + 1) * referee_dim,
                            symbol.answer * referee_dim : (symbol.answer + 1) * referee_dim,
                        ]
                        for a_ans in range(a_out)
                    )
                    constraints.append(current_block == sum_over_alice_outcomes)
            elif product_S_i_adj_S_j in seen_reduced_products:
                # This product S_k has been seen before as S_p^dagger S_q
                # So, Γ(S_i, S_j) = Γ(S_p, S_q)
                prev_i, prev_j = seen_reduced_products[product_S_i_adj_S_j]
                # Make sure to get the upper triangle element if current (i,j) is lower
                # The prev_i, prev_j should always refer to an upper triangle element by construction.
                previous_block = moment_matrix_R[
                    prev_i * referee_dim : (prev_i + 1) * referee_dim, prev_j * referee_dim : (prev_j + 1) * referee_dim
                ]
                constraints.append(current_block == previous_block)
            else:
                # First time seeing this operator product S_k
                seen_reduced_products[product_S_i_adj_S_j] = (i, j)

    # Constraints on the assemblage K_xy(a,b) itself --always apply all of these constraints!
    for x_alice_in in range(a_in):
        for y_bob_in in range(b_in):
            # Positivity: K_xy(a,b) >= 0 (operator PSD)
            for a_alice_out in range(a_out):
                for b_bob_out in range(b_out):
                    assemblage_block = assemblage[x_alice_in, y_bob_in][
                        a_alice_out * referee_dim : (a_alice_out + 1) * referee_dim,
                        b_bob_out * referee_dim : (b_bob_out + 1) * referee_dim,
                    ]
                    constraints.append(assemblage_block >> 0)

            # Normalization: Sum_{a,b} K_xy(a,b) = rho_R
            sum_over_outcomes_ab = sum(
                assemblage[x_alice_in, y_bob_in][
                    a * referee_dim : (a + 1) * referee_dim, b * referee_dim : (b + 1) * referee_dim
                ]
                for a in range(a_out)
                for b in range(b_out)
            )
            constraints.append(sum_over_outcomes_ab == rho_R_referee)
    if no_signaling:
        # No-signaling constraints on assemblage - ALWAYS APPLY
        # Bob's marginal rho_B(b|y) = Sum_a K_xy(a,b) must be independent of x
        for y_bob_in in range(b_in):
            for b_bob_out in range(b_out):
                sum_over_a_for_x0 = sum(
                    assemblage[0, y_bob_in][
                        a * referee_dim : (a + 1) * referee_dim, b_bob_out * referee_dim : (b_bob_out + 1) * referee_dim
                    ]
                    for a in range(a_out)
                )
                for x_alice_in in range(1, a_in):
                    sum_over_a_for_x_current = sum(
                        assemblage[x_alice_in, y_bob_in][
                            a * referee_dim : (a + 1) * referee_dim,
                            b_bob_out * referee_dim : (b_bob_out + 1) * referee_dim,
                        ]
                        for a in range(a_out)
                    )
                    constraints.append(sum_over_a_for_x0 == sum_over_a_for_x_current)

        # Alice's marginal rho_A(a|x) = Sum_b K_xy(a,b) must be independent of y
        for x_alice_in in range(a_in):
            for a_alice_out in range(a_out):  # For each Alice outcome a
                sum_over_b_for_y0 = sum(
                    assemblage[x_alice_in, 0][
                        a_alice_out * referee_dim : (a_alice_out + 1) * referee_dim,
                        b * referee_dim : (b + 1) * referee_dim,
                    ]
                    for b in range(b_out)
                )
                for y_bob_in in range(1, b_in):
                    sum_over_b_for_y_current = sum(
                        assemblage[x_alice_in, y_bob_in][
                            a_alice_out * referee_dim : (a_alice_out + 1) * referee_dim,
                            b * referee_dim : (b + 1) * referee_dim,
                        ]
                        for b in range(b_out)
                    )
                    constraints.append(sum_over_b_for_y0 == sum_over_b_for_y_current)

    return constraints
