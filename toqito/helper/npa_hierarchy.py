"""NPA constraints."""
from itertools import product
from collections import namedtuple
from typing import Dict, List, Set, Tuple, Union

import cvxpy


Symbol = namedtuple("Symbol", ["player", "question", "answer"], defaults=["", None, None])


# This function simplifies the input word by applying
# the commutation and projector rules.
def _reduce(word: Tuple[Symbol]) -> Tuple[Symbol]:
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


def _parse(k: str) -> Tuple[int, Set[Tuple[int, int]]]:
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
def _gen_words(
    k: Union[int, str], a_out: int, a_in: int, b_out: int, b_in: int
) -> List[Tuple[Symbol]]:
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


def _is_zero(word: Tuple[Symbol]) -> bool:
    return len(word) == 0


def _is_meas(word: Tuple[Symbol]) -> bool:
    if len(word) == 2:
        s_a, s_b = word
        return s_a.player == "Alice" and s_b.player == "Bob"

    return False


def _is_meas_on_one_player(word: Tuple[Symbol]) -> bool:
    if len(word) == 1 and word[0].player in ["Alice", "Bob"]:
        return True

    return False


def _get_shape(prob: Dict[Tuple[int, int], cvxpy.Variable]) -> Tuple[int, int, int, int]:
    a_in, a_out = (0, 0)
    b_in, b_out = (0, 0)
    for (x_in, y_in), _var in prob.items():
        a_in = max(a_in, x_in + 1)
        b_in = max(b_in, y_in + 1)
        a_out, b_out = _var.shape

    return a_out, a_in, b_out, b_in


def npa_constraints(
    prob: Dict[Tuple[int, int], cvxpy.Variable], k: Union[int, str] = 1
) -> List[cvxpy.constraints.constraint.Constraint]:
    """
    Generate the constraints specified by the NPA hierarchy up to a finite level.

    You can determine the level of the hierarchy by a positive integer or a string
    of a form like "1+ab+aab", which indicates that an intermediate level of the hierarchy
    should be used, where this example uses all products of 1 measurement, all products of
    one Alice and one Bob measurement, and all products of two Alice and one Bob measurement.

    The probabilities must be given as a dictionary. The keys are tuples of Alice and Bob questions
    and the values are cvxpy Variables representing the joint probability for different outcomes.

    :param prob: A dictionary with keys the different questions for Alice and Bob
                and values the probability matrices for different outcomes.
    :param k: The level of the NPA hierarchy to use (default=1).
    :return: A list of cvxpy constraints.
    """
    a_out, a_in, b_out, b_in = _get_shape(prob)

    words = _gen_words(k, a_out, a_in, b_out, b_in)
    dim = len(words)

    r_var = cvxpy.Variable((dim, dim), PSD=True, name="R")
    constraints = [r_var[0, 0] == 1]

    seen = {}
    for i in range(dim):
        for j in range(i, dim):
            w_i, w_j = words[i], words[j]
            w_i = tuple(reversed(w_i))
            word = _reduce(w_i + w_j)

            # if i = 0 we would consider (ε, ε) as an empty word.
            if i != 0 and _is_zero(word):
                constraints += [r_var[i, j] == 0]

            elif _is_meas(word):
                s_a, s_b = word
                constraints += [
                    r_var[i, j] == prob[s_a.question, s_b.question][s_a.answer, s_b.answer]
                ]

            elif _is_meas_on_one_player(word):
                symbol = word[0]
                if symbol.player == "Alice":
                    _sum = 0
                    for b_ans in range(b_out):
                        _sum += prob[symbol.question, 0][symbol.answer, b_ans]

                    constraints += [r_var[i, j] == _sum]

                if symbol.player == "Bob":
                    _sum = 0
                    for a_ans in range(a_out):
                        _sum += prob[0, symbol.question][a_ans, symbol.answer]

                    constraints += [r_var[i, j] == _sum]

            elif word in seen:
                old_i, old_j = seen[word]
                constraints += [r_var[i, j] == r_var[old_i, old_j]]

            else:
                seen[word] = (i, j)

    # now we impose constraints to the probability vector
    for x_alice_in in range(a_in):
        for y_bob_in in range(b_in):
            constraints += [
                prob[x_alice_in, y_bob_in] >= 0,
                cvxpy.sum(prob[x_alice_in, y_bob_in]) == 1,
            ]

    # Bob marginal consistency
    for y_bob_in in range(b_in):
        for b_ans in range(b_out):
            _sum = cvxpy.sum(prob[0, y_bob_in][:, b_ans])
            for x_alice_in in range(1, a_in):
                cur_sum = cvxpy.sum(prob[x_alice_in, y_bob_in][:, b_ans])
                constraints += [_sum == cur_sum]

    # Alice marginal consistency
    for x_alice_in in range(a_in):
        for a_ans in range(a_out):
            _sum = cvxpy.sum(prob[x_alice_in, 0][a_ans, :])
            for y_bob_in in range(1, b_in):
                cur_sum = cvxpy.sum(prob[x_alice_in, y_bob_in][a_ans, :])
                constraints += [_sum == cur_sum]

    return constraints
