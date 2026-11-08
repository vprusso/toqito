"""Generates the NPA constraints."""

from collections import namedtuple
from itertools import product

import cvxpy
import numpy as np
import scipy.sparse

Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])
IDENTITY_SYMBOL = Symbol()  # Identity symbol: the namedtuple defaults ("", None, None).
PLAYERS = ("Alice", "Bob")


def npa_constraints(
    assemblage: dict[tuple[int, int], cvxpy.Variable],
    k: int | str = 1,
    referee_dim: int = 1,
    no_signaling: bool = True,
    normalize: bool = True,
) -> list[cvxpy.constraints.constraint.Constraint]:
    r"""Generate the constraints specified by the NPA hierarchy up to a finite level.

    [@navascues2008convergent]

    You can determine the level of the hierarchy by a positive integer or a string
    of a form like "1+ab+aab", which indicates that an intermediate level of the hierarchy
    should be used, where this example uses all products of 1 measurement, all products of
    one Alice and one Bob measurement, and all products of two Alice and one Bob measurement.

    The commuting measurement assemblage operator must be given as a dictionary. The keys are
    tuples of Alice and Bob questions \(x, y\) and the values are cvxpy Variables which
    are matrices with entries:

    \[
        K_{xy}\Big(i + a \cdot dim_R, j + b \cdot dim_R \Big) =
        \langle i| \text{Tr}_{\mathcal{H}} \Big( \big(
            I_R \otimes A_a^x B_b^y \big) \sigma \Big) |j \rangle
    \]

    The moment matrix is parametrised by one free block per distinct operator product, so entries that must agree
    share a variable rather than being tied together by equality constraints, and the only constraints returned are
    positivity of the moment matrix, the identification of its low-order blocks with the assemblage, and the
    assemblage's own positivity, normalisation and (optionally) no-signaling conditions.

    Examples:
        The constraints returned here are meant to be handed straight to a cvxpy problem.
        Maximising the CHSH winning probability over the level-1 relaxation recovers the
        Tsirelson bound \(\cos^2(\pi/8) \approx 0.8536\), which is strictly above the
        classical value of \(0.75\).

        The CHSH game asks Alice and Bob a binary question each and accepts their binary
        answers when \(a \oplus b = x \land y\):

        ```python exec="1" source="above" result="text"
        import cvxpy

        from toqito.state_opt.npa_hierarchy import npa_constraints

        assemblage = {(x, y): cvxpy.Variable((2, 2), name=f"K({x},{y})") for x in range(2) for y in range(2)}

        p_win = 0
        for x in range(2):
            for y in range(2):
                for a in range(2):
                    for b in range(2):
                        if a ^ b == x * y:
                            p_win += 0.25 * assemblage[x, y][a, b]

        problem = cvxpy.Problem(cvxpy.Maximize(p_win), npa_constraints(assemblage, k=1))
        print(f"{problem.solve():.4f}")
        ```

    Args:
        assemblage: The commuting measurement assemblage operator.
        k: The level of the NPA hierarchy to use (default=1).
        referee_dim: The dimension of the referee's quantum system (default=1).
        no_signaling: Whether to impose the no-signaling conditions on the assemblage (default=True).
        normalize: Whether to fix the trace of the referee's state to one (default=True). Without it the constraints
            describe the cone of unnormalized assemblages, which is what a ratio of two linear functionals of the
            assemblage is maximized over.

    Returns:
        A list of cvxpy constraints.

    """
    a_out, a_in, b_out, b_in = _get_nonlocal_game_params(assemblage, referee_dim)

    words = _gen_words(k, a_out, a_in, b_out, b_in)
    if len(words) == 0:
        raise ValueError("Generated word list is empty. Check _gen_words logic.")

    # Moment matrix (Gamma matrix in [@navascues2008convergent]) with one `referee_dim`-dimensional block per pair of
    # words, generalised to referee systems as in extended nonlocal games [@speelman2016position].
    moment_matrix, first_position = _moment_matrix(words, referee_dim, complex_entries=referee_dim > 1)

    def block(i: int, j: int) -> cvxpy.Expression:
        return moment_matrix[i * referee_dim : (i + 1) * referee_dim, j * referee_dim : (j + 1) * referee_dim]

    def assemblage_block(x: int, y: int, a: int, b: int) -> cvxpy.Expression:
        return assemblage[x, y][a * referee_dim : (a + 1) * referee_dim, b * referee_dim : (b + 1) * referee_dim]

    # Sums of blocks over one player's answers are matrix products: `sum_a @ K_xy` is the row of blocks
    # sum_a K_xy(a, b) and `K_xy @ sum_b.T` the column of blocks sum_b K_xy(a, b).
    sum_a = np.kron(np.ones((1, a_out)), np.eye(referee_dim))
    sum_b = np.kron(np.ones((1, b_out)), np.eye(referee_dim))

    # The referee's effective state is the block of the identity word.
    rho_referee = block(*first_position[(IDENTITY_SYMBOL,)])
    constraints = [moment_matrix >> 0]
    if normalize:
        constraints.append(cvxpy.trace(rho_referee) == 1)

    # Identify the blocks of low-order words with the assemblage. Marginals of a single player are read off at the
    # other player's first question, which is immaterial under no-signaling.
    for word, (i, j) in first_position.items():
        if _is_meas(word):
            alice, bob = word
            target = assemblage_block(alice.question, bob.question, alice.answer, bob.answer)
        elif _is_meas_on_one_player(word):
            symbol = word[0]
            rows = slice(symbol.answer * referee_dim, (symbol.answer + 1) * referee_dim)
            if symbol.player == "Alice":
                target = (assemblage[symbol.question, 0] @ sum_b.T)[rows, :]
            else:
                target = (sum_a @ assemblage[0, symbol.question])[:, rows]
        else:
            continue
        constraints.append(block(i, j) == target)

    # Constraints on the assemblage itself: positivity of every block and normalisation to the referee's state.
    for x in range(a_in):
        for y in range(b_in):
            if referee_dim == 1:
                constraints.append(assemblage[x, y] >= 0)
            else:
                constraints += [assemblage_block(x, y, a, b) >> 0 for a in range(a_out) for b in range(b_out)]
            constraints.append(sum_a @ assemblage[x, y] @ sum_b.T == rho_referee)

    if no_signaling:
        # Bob's marginal sum_a K_xy(a, b) must not depend on x, and Alice's sum_b K_xy(a, b) must not depend on y.
        for y in range(b_in):
            reference = sum_a @ assemblage[0, y]
            constraints += [sum_a @ assemblage[x, y] == reference for x in range(1, a_in)]
        for x in range(a_in):
            reference = assemblage[x, 0] @ sum_b.T
            constraints += [assemblage[x, y] @ sum_b.T == reference for y in range(1, b_in)]

    return constraints


def bell_npa_constraints(
    p_var: cvxpy.Variable,
    desc: list[int],
    k: int | str = 1,
) -> list[cvxpy.constraints.constraint.Constraint]:
    r"""Generate NPA hierarchy constraints for Bell inequalities [@navascues2008convergent].

    The constraints are based on the positivity of a moment matrix constructed from measurement
    operators. This function generates constraints for a CVXPY variable representing probabilities
    or correlations in the Collins-Gisin notation. [@collins2004relevant]

    The level of the hierarchy ``k`` can be an integer (standard NPA level) or a string specifying
    intermediate levels (e.g., "1+ab", "2+aab").

    The input ``p_var`` is a CVXPY variable representing the probabilities in the Collins-Gisin (CG)
    notation. It should have dimensions \(((oa-1) \times ma+1, (ob-1) \times mb+1)\),
    where \(oa, ob\) are the number of outputs and \(ma, mb\) are the number of inputs for Alice
    and Bob, respectively, as specified in ``desc`` = [\(oa\), \(ob\), \(ma\), \(mb\)].
    The entries of ``p_var`` correspond to:
    - ``p_var[0, 0]``: The overall probability (should be 1).
    - ``p_var[i, 0]`` (for \(i > 0\)): Marginal probabilities/correlations for Alice.
    - ``p_var[0, j]`` (for \(j > 0\)): Marginal probabilities/correlations for Bob.
    - ``p_var[i, j]`` (for \(i > 0, j > 0\)): Joint probabilities/correlations for Alice and Bob.

    The mapping from indices \((i, j)\) to specific operators depends on the ordering defined by ``desc``.
    Specifically, if \(i = (oa-1) \times x + a + 1\) and \(j = (ob-1) \times y + b + 1\)

    - ``p_var[i, 0]`` corresponds to the expectation of Alice's operator
                      \(A_{a|x}\) (using \(0\) to \(oa-2\) for \(a\)).
    - ``p_var[0, j]`` corresponds to the expectation of Bob's operator
                      \(B_{b|y}\) (using \(0\) to \(ob-2\) for \(b\)).
    - ``p_var[i, j]`` corresponds to the expectation of the product \(A_{a|x} B_{b|y}\).

    The moment matrix is real and symmetric, parametrised by one free entry per distinct operator product, and the
    returned constraints are its positivity and a single vector equality identifying the low-order moments with the
    entries of ``p_var``.

    Args:
        p_var: A CVXPY Variable representing probabilities/correlations in Collins-Gisin notation.
                  Shape: \(((oa-1) \times ma+1, (ob-1) \times mb+1)\).
        desc: A list [\(oa\), \(ob\), \(ma\), \(mb\)]
                    specifying outputs and inputs for Alice and Bob.
        k: The level of the NPA hierarchy (integer or string like "1+ab"). Default is 1.


    Returns:
        A list of CVXPY constraints.

    Examples:
        Consider the CHSH inequality scenario with ``desc = [2, 2, 2, 2]``. We want to generate the NPA level 1
        constraints.

        ```python exec="1" source="above" result="text" session="npa_example"
        import cvxpy
        import numpy as np
        from toqito.state_opt.npa_hierarchy import bell_npa_constraints
        desc = [2, 2, 2, 2]
        oa, ob, ma, mb = desc
        p_var_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
        p_var = cvxpy.Variable(p_var_dim, name="p_cg")
        constraints = bell_npa_constraints(p_var, desc, k=1)
        print(len(constraints))
        print(constraints[0].args[0].shape)
        ```

        We can also use intermediate levels, like "1+ab":

        ```python exec="1" source="above" result="text" session="npa_example"
        constraints_1ab = bell_npa_constraints(p_var, desc, k="1+ab")
        print(constraints_1ab[0].args[0].shape)
        ```

        For the CGLMP inequality with ``dim=3``, ``desc = [3, 3, 2, 2]``, level 1:

        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.state_opt.npa_hierarchy import bell_npa_constraints
        desc_cglmp = [3, 3, 2, 2]
        oa_c, ob_c, ma_c, mb_c = desc_cglmp
        p_var_dim_c = ((oa_c - 1) * ma_c + 1, (ob_c - 1) * mb_c + 1)
        p_var_c = cvxpy.Variable(p_var_dim_c, name="p_cglmp")
        constraints_c = bell_npa_constraints(p_var_c, desc_cglmp, k=1)
        print(constraints_c[0].args[0].shape)
        ```


    """
    oa, ob, ma, mb = desc
    words = _gen_words(k, oa, ma, ob, mb)
    moment_matrix, first_position = _moment_matrix(words, referee_dim=1, complex_entries=False)

    # Every word with a Collins-Gisin entry is identified with it in one vector equality.
    moment_entries, cg_indices = [], []
    for word, (i, j) in first_position.items():
        cg_index = _word_to_p_cg_index(() if _is_identity(word) else word, oa, ob, ma, mb)
        if cg_index is not None:
            moment_entries.append(moment_matrix[i, j])
            cg_indices.append(cg_index)
    p_flat = p_var.flatten(order="F")
    return [moment_matrix >> 0, cvxpy.hstack(moment_entries) == p_flat[cg_indices]]


def _conjugate(word: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    """Return the reduced adjoint of a word.

    Projectors are self-adjoint, so the adjoint reverses the order of the factors; reducing restores the canonical
    Alice-before-Bob order, which reverses each player's block separately.
    """
    return _reduce(word[::-1])


def _word_key(word: tuple[Symbol, ...]) -> tuple:
    """Return a sortable key for a word, used to order words and to pick class representatives."""
    return (len(word), tuple(repr(symbol) for symbol in word))


def _moment_matrix(
    words: list[tuple[Symbol, ...]], referee_dim: int, complex_entries: bool
) -> tuple[cvxpy.Expression, dict[tuple[Symbol, ...], tuple[int, int]]]:
    r"""Build the moment matrix of a word list as a linear image of free variables.

    Block ``(i, j)`` of the moment matrix is the moment of the product \(S_i^\dagger S_j\). Products that reduce to
    the same word, or to adjoint words, share one ``referee_dim``-dimensional block variable, with the adjoint word
    carrying the block's conjugate transpose, so the matrix is Hermitian (symmetric in the real case) by
    construction. Products that reduce to zero are fixed to zero.

    Returns the matrix expression and, for every reduced word, the first ``(i, j)`` block at which it appears.
    """
    n = len(words)
    dim = referee_dim
    # Each class is an unordered pair {word, adjoint}; `representatives[word]` names it.
    representatives: dict[tuple[Symbol, ...], tuple[Symbol, ...]] = {}
    first_position: dict[tuple[Symbol, ...], tuple[int, int]] = {}
    # One entry per (block position, block row, block column): the variable index and sign for the real part, and
    # for the imaginary part when the matrix is complex.
    placements: list[tuple[int, int, tuple[Symbol, ...], bool]] = []
    for i in range(n):
        for j in range(n):
            word = _reduce(words[i][::-1] + words[j])
            if _is_zero(word):
                continue
            first_position.setdefault(word, (i, j))
            adjoint = _conjugate(word)
            representative = min(word, adjoint, key=_word_key)
            representatives.setdefault(word, representative)
            placements.append((i, j, representative, word != representative))

    classes = sorted(set(representatives.values()), key=_word_key)
    # Variable layout: for each class, a `dim`-by-`dim` block. A self-adjoint class holds a Hermitian block, so only
    # the upper triangle of its real part and the strict upper triangle of its imaginary part are free; any other
    # class holds a general block.
    real_index: dict[tuple, int] = {}
    imag_index: dict[tuple, int] = {}
    for representative in classes:
        self_adjoint = _conjugate(representative) == representative
        for r in range(dim):
            for s in range(dim):
                if self_adjoint and r > s:
                    continue
                real_index[representative, r, s] = len(real_index)
                if complex_entries and not (self_adjoint and r == s):
                    imag_index[representative, r, s] = len(imag_index)

    def lookup(representative: tuple[Symbol, ...], r: int, s: int) -> tuple[int, int | None, float]:
        """Return the real variable index, the imaginary variable index and its sign for entry (r, s) of a block."""
        if (representative, r, s) in real_index:
            return real_index[representative, r, s], imag_index.get((representative, r, s)), 1.0
        # Lower triangle of a Hermitian block: the conjugate of the mirrored entry.
        return real_index[representative, s, r], imag_index.get((representative, s, r)), -1.0

    rows, real_cols, imag_rows, imag_cols, imag_signs = [], [], [], [], []
    for i, j, representative, is_adjoint in placements:
        for r in range(dim):
            for s in range(dim):
                # The adjoint word carries the conjugate transpose of the block.
                re_idx, im_idx, sign = lookup(representative, s, r) if is_adjoint else lookup(representative, r, s)
                entry = (i * dim + r) * (n * dim) + (j * dim + s)
                rows.append(entry)
                real_cols.append(re_idx)
                if im_idx is not None:
                    imag_rows.append(entry)
                    imag_cols.append(im_idx)
                    imag_signs.append(-sign if is_adjoint else sign)

    size = n * dim
    real_map = scipy.sparse.csc_matrix((np.ones(len(rows)), (rows, real_cols)), shape=(size * size, len(real_index)))
    real_part = cvxpy.Variable(len(real_index), name="moments")
    moment_matrix = cvxpy.reshape(real_map @ real_part, (size, size), order="C")
    if imag_index:
        imag_map = scipy.sparse.csc_matrix((imag_signs, (imag_rows, imag_cols)), shape=(size * size, len(imag_index)))
        imag_part = cvxpy.Variable(len(imag_index), name="moments_imag")
        moment_matrix = moment_matrix + 1j * cvxpy.reshape(imag_map @ imag_part, (size, size), order="C")
    return moment_matrix, first_position


def _reduce(word: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    """Reduce an operator word to its canonical form using NPA rules.

    Identity: I*S = S*I = S, I*I = I
    Commutation: Alice operators commute with Bob operators. Canonical form: A...AB...B
    Orthogonality: P_x,a P_x,b = 0 if a != b (for same player x)
    Idempotence: P_x,a P_x,a = P_x,a (for same player x)

    The empty tuple represents the zero operator and ``(IDENTITY_SYMBOL,)`` the identity.

    """
    if not word:
        return ()
    symbols = [symbol for symbol in word if symbol != IDENTITY_SYMBOL]
    if not symbols:
        return (IDENTITY_SYMBOL,)

    reduced: list[Symbol] = []
    # Alice's operators commute with Bob's, so the canonical order is all of Alice's followed by all of Bob's.
    for symbol in [s for s in symbols if s.player == "Alice"] + [s for s in symbols if s.player == "Bob"]:
        if reduced and reduced[-1] == symbol:
            continue  # Idempotence.
        if reduced and reduced[-1].player == symbol.player and reduced[-1].question == symbol.question:
            return ()  # Orthogonal projectors of one measurement.
        reduced.append(symbol)
    return tuple(reduced)


def _parse(k_str: str) -> tuple[int, set[tuple[int, int]]]:
    """Parse a level string such as ``"1+ab+aab"`` into the base level and the extra (Alice, Bob) length pairs."""
    if not k_str:
        raise ValueError("Input string k_str cannot be empty.")
    parts = k_str.split("+")
    if not parts[0]:
        raise ValueError("Base level k must be specified, e.g., '1+ab'")
    try:
        base_k = int(parts[0])
    except ValueError as e:
        raise ValueError(f"Base level k '{parts[0]}' is not a valid integer: {e}") from e

    configurations = set()
    for component in parts[1:]:
        if not component:
            continue
        for char in component:
            if char not in ("a", "b"):
                raise ValueError(
                    f"Invalid character '{char}' in k string component '{component}'. "
                    "Only 'a' or 'b' allowed after base k."
                )
        configurations.add((component.count("a"), component.count("b")))
    return base_k, configurations


def _gen_words(k: int | str, a_out: int, a_in: int, b_out: int, b_in: int) -> list[tuple[Symbol, ...]]:
    """Generate the reduced words indexing the moment matrix at level ``k``, with the identity first."""
    # Symbols for non-identity measurements (the last outcome of each measurement is dependent).
    alice_symbols = [Symbol("Alice", x, a) for x in range(a_in) for a in range(a_out - 1)]
    bob_symbols = [Symbol("Bob", y, b) for y in range(b_in) for b in range(b_out - 1)]

    base_k, configurations = _parse(k) if isinstance(k, str) else (k, set())
    # Every (Alice length, Bob length) pair up to the base level, plus the extra configurations.
    length_pairs = {(alice_len, length - alice_len) for length in range(base_k + 1) for alice_len in range(length + 1)}
    length_pairs |= configurations

    words = {(IDENTITY_SYMBOL,)}
    for alice_len, bob_len in length_pairs:
        for alice_word in product(alice_symbols, repeat=alice_len):
            reduced_alice = _reduce(alice_word)
            if _is_zero(reduced_alice) and alice_len > 0:
                continue
            for bob_word in product(bob_symbols, repeat=bob_len):
                reduced_bob = _reduce(bob_word)
                if _is_zero(reduced_bob) and bob_len > 0:
                    continue
                words.add(_reduce(reduced_alice + reduced_bob))

    words.discard(())
    words.discard((IDENTITY_SYMBOL,))
    return [(IDENTITY_SYMBOL,)] + sorted(words, key=_word_key)


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


def _word_to_p_cg_index(word: tuple[Symbol, ...], oa: int, ob: int, ma: int, mb: int) -> int | None:
    """Map an operator word to its corresponding index in the flattened CG vector."""
    dim_a = (oa - 1) * ma
    dim_b = (ob - 1) * mb
    row_dim = dim_a + 1
    col_dim = dim_b + 1
    order = "F"
    if not word:
        return 0
    if len(word) == 1:
        s = word[0]
        if s.player == "Alice":
            row_idx = (oa - 1) * s.question + s.answer + 1
            return np.ravel_multi_index((row_idx, 0), (row_dim, col_dim), order=order)
        if s.player == "Bob":
            col_idx = (ob - 1) * s.question + s.answer + 1
            return np.ravel_multi_index((0, col_idx), (row_dim, col_dim), order=order)
    if len(word) == 2:
        s_a, s_b = word
        if s_a.player == "Alice" and s_b.player == "Bob":
            row_idx = (oa - 1) * s_a.question + s_a.answer + 1
            col_idx = (ob - 1) * s_b.question + s_b.answer + 1
            return np.ravel_multi_index((row_idx, col_idx), (row_dim, col_dim), order=order)
    return None
