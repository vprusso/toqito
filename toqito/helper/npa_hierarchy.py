"""NPA constraints."""


from collections import namedtuple
from itertools import product

import cvxpy

Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])


# This function simplifies the input word by applying
# the commutation and projector rules.
def _reduce(word: tuple[Symbol]) -> tuple[Symbol]:
    # commute: bring Alice in front.
    w_a, w_b = (), ()
    for symbol in word:
        if symbol.player == "Alice":
            w_a += (symbol,)
        if symbol.player == "Bob":
            w_b += (symbol,)

    word = w_a + w_b
    for i in range(len(word) - 1):
        symbol_x, symbol_y = word[i], word[i + 1]

        # projector: merge them.
        if symbol_x == symbol_y:
            return _reduce(word[:i] + word[i + 1 :])

        # orthogonal: evaluates to zero.
        if (
            symbol_x.player == symbol_y.player
            and symbol_x.question == symbol_y.question
            and symbol_x.answer != symbol_y.answer
        ):
            return ()

    return word


def _parse(k: str) -> tuple[int, set[tuple[int, int]]]:
    k = k.split("+")
    base_k = int(k[0])

    conf = set()
    for val in k[1:]:
        # otherwise we already take this configuration
        # in base_k - level of hierarchy.
        if len(val) > base_k:
            cnt_a, cnt_b = 0, 0
            for bit in val:
                if bit == "a":
                    cnt_a += 1
                if bit == "b":
                    cnt_b += 1

            conf.add((cnt_a, cnt_b))

    return base_k, conf


# This function generates all non - equivalent words of length up to k.
def _gen_words(k: int | str, a_out: int, a_in: int, b_out: int, b_in: int) -> list[tuple[Symbol]]:
    # remove one outcome to avoid redundancy
    # since all projectors sum to identity.
    b_symbols = [Symbol("Bob", y, b) for y in range(b_in) for b in range(b_out - 1)]
    a_symbols = [Symbol("Alice", x, a) for x in range(a_in) for a in range(a_out - 1)]

    words = [(Symbol(""),)]

    conf = []
    if isinstance(k, str):
        k, conf = _parse(k)

    # pylint: disable=too-many-nested-blocks
    for i in range(1, k + 1):
        for j in range(i + 1):
            # words of type: a^j b^(i - j)
            for word_a in product(a_symbols, repeat=j):
                if len(_reduce(word_a)) == j:
                    for word_b in product(b_symbols, repeat=i - j):
                        if len(_reduce(word_b)) == i - j:
                            words += [word_a + word_b]

    # now generate the intermediate levels of hierarchy
    for cnt_a, cnt_b in conf:
        for word_a in product(a_symbols, repeat=cnt_a):
            if len(_reduce(word_a)) == cnt_a:
                for word_b in product(b_symbols, repeat=cnt_b):
                    if len(_reduce(word_b)) == cnt_b:
                        words += [word_a + word_b]

    return words


def _is_zero(word: tuple[Symbol]) -> bool:
    return len(word) == 0


def _is_meas(word: tuple[Symbol]) -> bool:
    if len(word) == 2:
        s_a, s_b = word
        return s_a.player == "Alice" and s_b.player == "Bob"

    return False


def _is_meas_on_one_player(word: tuple[Symbol]) -> bool:
    return len(word) == 1 and word[0].player in {"Alice", "Bob"}


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


def npa_constraints(  # pylint: disable=too-many-locals
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

    r_var = cvxpy.Variable((referee_dim * dim, referee_dim * dim), PSD=True, name="R")
    # Normalization.
    norm = sum(r_var[i * dim, i * dim] for i in range(referee_dim))
    constraints = [norm == 1]

    seen = {}
    for i in range(dim):
        for j in range(i, dim):
            w_i, w_j = words[i], words[j]
            w_i = tuple(reversed(w_i))
            word = _reduce(w_i + w_j)

            sub_mat = r_var[i::dim, j::dim]
            # if i = 0 we would consider (ε, ε) as an empty word.
            if i != 0 and _is_zero(word):
                constraints.append(sub_mat == 0)

            elif _is_meas(word):
                s_a, s_b = word
                constraints.append(
                    sub_mat
                    == assemblage[s_a.question, s_b.question][
                        s_a.answer * referee_dim : (s_a.answer + 1) * referee_dim,
                        s_b.answer * referee_dim : (s_b.answer + 1) * referee_dim,
                    ]
                )

            elif _is_meas_on_one_player(word):
                symbol = word[0]
                if symbol.player == "Alice":
                    sum_all_bob_meas = sum(
                        assemblage[symbol.question, 0][
                            symbol.answer * referee_dim : (symbol.answer + 1) * referee_dim,
                            b_ans * referee_dim : (b_ans + 1) * referee_dim,
                        ]
                        for b_ans in range(b_out)
                    )

                    constraints.append(sub_mat == sum_all_bob_meas)

                if symbol.player == "Bob":
                    sum_all_alice_meas = sum(
                        assemblage[0, symbol.question][
                            a_ans * referee_dim : (a_ans + 1) * referee_dim,
                            symbol.answer * referee_dim : (symbol.answer + 1) * referee_dim,
                        ]
                        for a_ans in range(a_out)
                    )

                    constraints.append(sub_mat == sum_all_alice_meas)

            elif word in seen:
                old_i, old_j = seen[word]
                old_sub_mat = r_var[old_i::dim, old_j::dim]
                constraints.append(sub_mat == old_sub_mat)

            else:
                seen[word] = (i, j)

    # now we impose constraints to the assemblage operator
    for x_alice_in in range(a_in):
        for y_bob_in in range(b_in):
            sum_all_meas_and_trace = 0
            for a_ans in range(a_out):
                for b_ans in range(b_out):
                    sum_all_meas_and_trace += sum(
                        assemblage[x_alice_in, y_bob_in][
                            i + a_ans * referee_dim, i + b_ans * referee_dim
                        ]
                        for i in range(referee_dim)
                    )

                    # r x r sub - block is PSD since it's an unnormalized quantum state.
                    constraints.append(
                        assemblage[x_alice_in, y_bob_in][
                            a_ans * referee_dim : (a_ans + 1) * referee_dim,
                            b_ans * referee_dim : (b_ans + 1) * referee_dim,
                        ]
                        >> 0
                    )

            constraints.append(sum_all_meas_and_trace == 1)

    # Bob marginal consistency
    for y_bob_in in range(b_in):
        for b_ans in range(b_out):
            sum_first_question = sum(
                assemblage[0, y_bob_in][
                    a_ans * referee_dim : (a_ans + 1) * referee_dim,
                    b_ans * referee_dim : (b_ans + 1) * referee_dim,
                ]
                for a_ans in range(a_out)
            )

            for x_alice_in in range(1, a_in):
                sum_cur_question = sum(
                    assemblage[x_alice_in, y_bob_in][
                        a_ans * referee_dim : (a_ans + 1) * referee_dim,
                        b_ans * referee_dim : (b_ans + 1) * referee_dim,
                    ]
                    for a_ans in range(a_out)
                )

                constraints.append(sum_first_question == sum_cur_question)

    # Alice marginal consistency
    for x_alice_in in range(a_in):
        for a_ans in range(a_out):
            sum_first_question = sum(
                assemblage[x_alice_in, 0][
                    a_ans * referee_dim : (a_ans + 1) * referee_dim,
                    b_ans * referee_dim : (b_ans + 1) * referee_dim,
                ]
                for b_ans in range(b_out)
            )

            for y_bob_in in range(1, b_in):
                sum_cur_question = sum(
                    assemblage[x_alice_in, y_bob_in][
                        a_ans * referee_dim : (a_ans + 1) * referee_dim,
                        b_ans * referee_dim : (b_ans + 1) * referee_dim,
                    ]
                    for b_ans in range(b_out)
                )

                constraints.append(sum_first_question == sum_cur_question)

    return constraints
