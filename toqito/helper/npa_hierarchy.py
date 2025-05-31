"""Generates the NPA constraints."""

from collections import namedtuple
from itertools import product

import cvxpy

Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])
IDENTITY_SYMBOL = Symbol("", None, None)  # Explicit identity symbol


def _reduce(word: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    if not word:
        return ()
    if word == (IDENTITY_SYMBOL,):  # Identity reduces to identity
        return (IDENTITY_SYMBOL,)

    # Separate Alice and Bob symbols
    w_a = tuple(s for s in word if s.player == "Alice")
    w_b = tuple(s for s in word if s.player == "Bob")

    # Reduce Alice's part
    reduced_w_a = []
    if w_a:
        reduced_w_a.append(w_a[0])
        for i in range(1, len(w_a)):
            # Orthogonality: P_i P_j = 0 if i != j (for same question)
            if w_a[i].question == reduced_w_a[-1].question and w_a[i].answer != reduced_w_a[-1].answer:
                return ()  # Product is zero
            # Idempotence: P_i P_i = P_i
            if w_a[i] != reduced_w_a[-1]:
                reduced_w_a.append(w_a[i])

    # Reduce Bob's part
    reduced_w_b = []
    if w_b:
        reduced_w_b.append(w_b[0])
        for i in range(1, len(w_b)):
            if w_b[i].question == reduced_w_b[-1].question and w_b[i].answer != reduced_w_b[-1].answer:
                return ()  # Product is zero
            if w_b[i] != reduced_w_b[-1]:
                reduced_w_b.append(w_b[i])

    final_word = tuple(reduced_w_a) + tuple(reduced_w_b)
    if not final_word:  # If all symbols got cancelled (e.g. A0A1 B0B1)
        return ()
    if final_word == (IDENTITY_SYMBOL,) and (
        tuple(reduced_w_a) == (IDENTITY_SYMBOL,) or tuple(reduced_w_b) == (IDENTITY_SYMBOL,)
    ):
        # Avoid ((ID), B) becoming (ID, B) then just (ID) if B was empty.
        # If the only non-empty part was identity, it's just identity.
        # This logic might need more refinement depending on how Identity is used in product.
        # For now, if it's just identity, it's identity.
        pass

    # A more direct approach for single identity: if the word was (ID_SYMBOL,) it should remain so.
    # The logic above might make reduced_w_a or reduced_w_b empty if original only had ID.
    # The initial `words = set([(IDENTITY_SYMBOL,)])` handles the pure identity.
    # A product like A_0 * I should be A_0. _reduce( (A_0, I) )
    # The current _reduce doesn't explicitly handle identity symbol in products.
    # Let's assume identity symbols are filtered out before forming composite words,
    # or _reduce is only called on sequences of actual measurement operators.
    # The current _gen_words builds words from a_symbols, b_symbols which don't include IDENTITY_SYMBOL.
    # The only (IDENTITY_SYMBOL,) comes from the initial set.

    # If after reduction, the word is empty, it means it's an algebraic zero.
    if not final_word:
        return ()
    return final_word


def _parse(k_str: str) -> tuple[int, set[tuple[int, int]]]:
    parts = k_str.split("+")
    if not parts[0]:  # Handle cases like "+ab" if they were allowed (they are not currently)
        raise ValueError("Base level k must be specified, e.g., '1+ab'")
    base_k = int(parts[0])
    conf = set()
    for val in parts[1:]:
        cnt_a, cnt_b = 0, 0
        if not val:
            continue  # Skip empty strings if k_str is like "1+"
        for char_val in val:
            if char_val == "a":
                cnt_a += 1
            elif char_val == "b":
                cnt_b += 1
            else:
                raise ValueError(f"Invalid character '{char_val}' in k string component '{val}'")
        if cnt_a > 0 or cnt_b > 0:
            conf.add((cnt_a, cnt_b))
    return base_k, conf


def _gen_words(k: int | str, a_out: int, a_in: int, b_out: int, b_in: int) -> list[tuple[Symbol, ...]]:
    # Symbols for non-identity measurements (last outcome is dependent)
    alice_symbols = [Symbol("Alice", x, a) for x in range(a_in) for a in range(a_out - 1)]
    bob_symbols = [Symbol("Bob", y, b) for y in range(b_in) for b in range(b_out - 1)]

    words = set([(IDENTITY_SYMBOL,)])  # Start with identity operator

    k_int = k
    configurations = set()  # Additional (length_A, length_B) configurations

    if isinstance(k, str):
        k_int, configurations = _parse(k)

    # Generate words up to length k_int
    for length in range(1, k_int + 1):
        for alice_len in range(length + 1):
            bob_len = length - alice_len

            for word_a_tuple in product(alice_symbols, repeat=alice_len):
                reduced_a = _reduce(word_a_tuple)
                if reduced_a == () and alice_len > 0:
                    continue  # Skip if Alice's part is zero

                for word_b_tuple in product(bob_symbols, repeat=bob_len):
                    reduced_b = _reduce(word_b_tuple)
                    if reduced_b == () and bob_len > 0:
                        continue  # Skip if Bob's part is zero

                    # Construct combined word: Alice's part then Bob's part
                    # _reduce on combined (Alice_reduced, Bob_reduced) handles commutation
                    # and potential further reductions if, e.g., A and B parts were empty.
                    combined_word_unreduced = (reduced_a if reduced_a != (IDENTITY_SYMBOL,) else ()) + (
                        reduced_b if reduced_b != (IDENTITY_SYMBOL,) else ()
                    )

                    if not combined_word_unreduced:  # if both parts reduced to identity or one was zero
                        if reduced_a == () or reduced_b == ():  # one part was zero
                            # This case implies the whole product is zero, _reduce will handle it
                            pass  # Let _reduce on combined handle it
                        else:  # both were identity or empty
                            words.add((IDENTITY_SYMBOL,))  # Ensure identity is added if it's the result
                            continue

                    final_word = _reduce(combined_word_unreduced)
                    if final_word:  # Add if not algebraically zero
                        words.add(final_word)

    # Add words from specific configurations (e.g., "1+ab")
    for alice_len, bob_len in configurations:
        if alice_len == 0 and bob_len == 0:
            continue  # Already have identity

        for word_a_tuple in product(alice_symbols, repeat=alice_len):
            reduced_a = _reduce(word_a_tuple)
            if reduced_a == () and alice_len > 0:
                continue

            for word_b_tuple in product(bob_symbols, repeat=bob_len):
                reduced_b = _reduce(word_b_tuple)
                if reduced_b == () and bob_len > 0:
                    continue

                combined_word_unreduced = (reduced_a if reduced_a != (IDENTITY_SYMBOL,) else ()) + (
                    reduced_b if reduced_b != (IDENTITY_SYMBOL,) else ()
                )
                if not combined_word_unreduced:
                    if reduced_a == () or reduced_b == ():
                        pass
                    else:
                        words.add((IDENTITY_SYMBOL,))
                        continue

                final_word = _reduce(combined_word_unreduced)
                if final_word:
                    words.add(final_word)

    # Sort for consistent ordering, important for matrix indexing
    # Identity first, then by length, then lexicographically.
    # return sorted(list(words), key=lambda w: (len(w), w)) # Previous sorting
    # Ensure IDENTITY_SYMBOL is first
    sorted_words = sorted([w for w in words if w != (IDENTITY_SYMBOL,)], key=lambda wd: (len(wd), wd))
    return [(IDENTITY_SYMBOL,)] + sorted_words


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
        return s.player in ("Alice", "Bob")  # Excludes IDENTITY_SYMBOL
    return False


# _get_nonlocal_game_params remains the same as in npa_constraints_fix
def _get_nonlocal_game_params(
    assemblage: dict[tuple[int, int], cvxpy.Variable], referee_dim: int = 1
) -> tuple[int, int, int, int]:
    a_in, b_in = max(assemblage.keys())
    a_in = a_in + 1
    b_in = b_in + 1
    operator = next(iter(assemblage.values()))
    a_out = int(operator.shape[0] / referee_dim)
    b_out = int(operator.shape[1] / referee_dim)
    return a_out, a_in, b_out, b_in


def npa_constraints(
    assemblage: dict[tuple[int, int], cvxpy.Variable], k: int | str = 1, referee_dim: int = 1
) -> list[cvxpy.constraints.constraint.Constraint]:
    r"""Generate the constraints specified by the NPA hierarchy up to a finite level :cite:`Navascues_2008_AConvergent`.

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
    .. bibliography::
        :filter: docname in docnames

    :param assemblage: The commuting measurement assemblage operator.
    :param k: The level of the NPA hierarchy to use (default=1).
    :param referee_dim: The dimension of the referee's quantum system (default=1).
    :return: A list of cvxpy constraints.

    """
    a_out, a_in, b_out, b_in = _get_nonlocal_game_params(assemblage, referee_dim)

    words = _gen_words(k, a_out, a_in, b_out, b_in)
    dim = len(words)

    # --- Debugging Print ---
    print(f"NPA level k='{k}', referee_dim={referee_dim}")
    print(f"Calculated params: a_out={a_out}, a_in={a_in}, b_out={b_out}, b_in={b_in}")
    print(f"Number of words (dim for moment matrix): {dim}")
    # print(f"Generated words: {words}")
    # --- End Debugging Print ---

    if dim == 0:  # Should not happen if IDENTITY_SYMBOL is always included
        raise ValueError("Generated word list is empty. Check _gen_words logic.")

    # Moment matrix (Gamma matrix in NPA paper)
    # r_var[i,j] block corresponds to E[S_i^dagger S_j]
    moment_matrix_R = cvxpy.Variable((referee_dim * dim, referee_dim * dim), hermitian=True, name="R")

    # Referee's effective state rho_R = E[I]
    # This is the (0,0) block of moment_matrix_R since words[0] is Identity
    rho_R_referee = moment_matrix_R[0:referee_dim, 0:referee_dim]

    constraints = [
        cvxpy.trace(rho_R_referee) == 1,  # Tr(rho_R) = 1
        rho_R_referee >> 0,  # rho_R is PSD
        moment_matrix_R >> 0,  # Entire moment matrix is PSD
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

            if not product_unreduced:  # This happens if both words[i] and words[j] were IDENTITY_SYMBOL
                product_S_i_adj_S_j = (IDENTITY_SYMBOL,)
            else:
                product_S_i_adj_S_j = _reduce(tuple(product_unreduced))

            current_block = moment_matrix_R[
                i * referee_dim : (i + 1) * referee_dim, j * referee_dim : (j + 1) * referee_dim
            ]

            if _is_zero(product_S_i_adj_S_j):  # Product is algebraically zero
                constraints.append(current_block == 0)
            elif _is_identity(product_S_i_adj_S_j):  # Product is identity operator
                # This occurs for (i,j) where S_i^dagger S_j = I. e.g. S_i = S_j and S_i is unitary (proj).
                # Or i=0, j=0 (I^dagger I = I).
                # This means current_block should be rho_R_referee if product_S_i_adj_S_j is I
                if i == 0 and j == 0:  # This is rho_R itself, already handled by definition
                    pass
                else:  # For other S_i^dagger S_j = I, their block should also be rho_R
                    constraints.append(current_block == rho_R_referee)

            elif _is_meas(product_S_i_adj_S_j):  # Product is A_a^x B_b^y
                alice_symbol, bob_symbol = product_S_i_adj_S_j
                constraints.append(
                    current_block
                    == assemblage[alice_symbol.question, bob_symbol.question][
                        alice_symbol.answer * referee_dim : (alice_symbol.answer + 1) * referee_dim,
                        bob_symbol.answer * referee_dim : (bob_symbol.answer + 1) * referee_dim,
                    ]
                )
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
                elif symbol.player == "Bob":
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

    # Constraints on the assemblage K_xy(a,b) itself - ALWAYS APPLY ALL OF THESE
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

    # No-signaling constraints on assemblage - ALWAYS APPLY
    # Bob's marginal rho_B(b|y) = Sum_a K_xy(a,b) must be independent of x
    for y_bob_in in range(b_in):
        for b_bob_out in range(b_out):  # For each Bob outcome b
            sum_over_a_for_x0 = sum(
                assemblage[0, y_bob_in][
                    a * referee_dim : (a + 1) * referee_dim, b_bob_out * referee_dim : (b_bob_out + 1) * referee_dim
                ]
                for a in range(a_out)
            )
            for x_alice_in in range(1, a_in):
                sum_over_a_for_x_current = sum(
                    assemblage[x_alice_in, y_bob_in][
                        a * referee_dim : (a + 1) * referee_dim, b_bob_out * referee_dim : (b_bob_out + 1) * referee_dim
                    ]
                    for a in range(a_out)
                )
                constraints.append(sum_over_a_for_x0 == sum_over_a_for_x_current)

    # Alice's marginal rho_A(a|x) = Sum_b K_xy(a,b) must be independent of y
    for x_alice_in in range(a_in):
        for a_alice_out in range(a_out):  # For each Alice outcome a
            sum_over_b_for_y0 = sum(
                assemblage[x_alice_in, 0][
                    a_alice_out * referee_dim : (a_alice_out + 1) * referee_dim, b * referee_dim : (b + 1) * referee_dim
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
