"""Generates NPA hierarchy constraints for Bell inequalities."""

from typing import List, Tuple, Union

import cvxpy
import numpy as np

from toqito.helper.npa_hierarchy import Symbol, _gen_words, _reduce


def _word_to_p_cg_index(word: Tuple[Symbol, ...], oa: int, ob: int, ma: int, mb: int) -> Union[int, None]:
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


def bell_npa_constraints(
    p_var: cvxpy.Variable,
    desc: List[int],
    k: Union[int, str] = 1,
) -> List[cvxpy.constraints.constraint.Constraint]:
    r"""Generate NPA hierarchy constraints for Bell inequalities :cite:`Navascues_2008_AConvergent`.

    The constraints are based on the positivity of a moment matrix constructed from measurement
    operators. This function generates constraints for a CVXPY variable representing probabilities
    or correlations in the Collins-Gisin notation.

    The level of the hierarchy `k` can be an integer (standard NPA level) or a string specifying
    intermediate levels (e.g., "1+ab", "2+aab").

    The input `p_var` is a CVXPY variable representing the probabilities in the Collins-Gisin (CG)
    notation. It should have dimensions `((oa-1)*ma+1, (ob-1)*mb+1)`, where `oa, ob` are the number
    of outputs and `ma, mb` are the number of inputs for Alice and Bob, respectively, as specified
    in `desc = [oa, ob, ma, mb]`. The entries of `p_var` correspond to:
    - `p_var[0, 0]`: The overall probability (should be 1).
    - `p_var[i, 0]` (for i > 0): Marginal probabilities/correlations for Alice.
    - `p_var[0, j]` (for j > 0): Marginal probabilities/correlations for Bob.
    - `p_var[i, j]` (for i > 0, j > 0): Joint probabilities/correlations for Alice and Bob.

    The mapping from indices `(i, j)` to specific operators depends on the ordering defined by `desc`.
    Specifically, if `i = (oa-1)*x + a + 1` and `j = (ob-1)*y + b + 1`:
    - `p_var[i, 0]` corresponds to the expectation of Alice's operator `A_{a|x}` (using 0 to `oa-2` for `a`).
    - `p_var[0, j]` corresponds to the expectation of Bob's operator `B_{b|y}` (using 0 to `ob-2` for `b`).
    - `p_var[i, j]` corresponds to the expectation of the product `A_{a|x} B_{b|y}`.

    Examples
    ========

    Consider the CHSH inequality scenario with `desc = [2, 2, 2, 2]`. We want to generate the NPA level 1 constraints.

    >>> import cvxpy
    >>> import numpy as np
    >>> from toqito.helper import bell_npa_constraints
    >>> desc = [2, 2, 2, 2]
    >>> oa, ob, ma, mb = desc
    >>> p_var_dim = ((oa - 1) * ma + 1, (ob - 1) * mb + 1)
    >>> p_var = cvxpy.Variable(p_var_dim, name="p_cg")
    >>> constraints = bell_npa_constraints(p_var, desc, k=1)
    >>> print(len(constraints))
    14
    >>> print(constraints[0])
    Gamma + Promote(-0.0, (5, 5)) >> 0

    We can also use intermediate levels, like "1+ab":

    >>> constraints_1ab = bell_npa_constraints(p_var, desc, k="1+ab")
    >>> print(len(constraints_1ab))
    34
    >>> print(constraints_1ab[0])
    Gamma + Promote(-0.0, (9, 9)) >> 0

    For the CGLMP inequality with `dim=3`, `desc = [3, 3, 2, 2]`, level 1:

    >>> desc_cglmp = [3, 3, 2, 2]
    >>> oa_c, ob_c, ma_c, mb_c = desc_cglmp
    >>> p_var_dim_c = ((oa_c - 1) * ma_c + 1, (ob_c - 1) * mb_c + 1)
    >>> p_var_c = cvxpy.Variable(p_var_dim_c, name="p_cglmp")
    >>> constraints_c = bell_npa_constraints(p_var_c, desc_cglmp, k=1)
    >>> print(len(constraints_c))
    38
    >>> print(constraints_c[0])
    Gamma + Promote(-0.0, (9, 9)) >> 0

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param p_var: A CVXPY Variable representing probabilities/correlations in Collins-Gisin notation.
                  Shape: `((oa-1)*ma+1, (ob-1)*mb+1)`.
    :param desc: A list `[oa, ob, ma, mb]` specifying outputs and inputs for Alice and Bob.
    :param k: The level of the NPA hierarchy (integer or string like "1+ab"). Default is 1.
    :return: A list of CVXPY constraints.
    :raises ValueError: If internal identity mapping fails.

    """
    oa, ob, ma, mb = desc
    words = _gen_words(k, oa, ma, ob, mb)
    dim = len(words)
    gamma = cvxpy.Variable((dim, dim), name="Gamma", symmetric=True)
    constraints = [gamma >> 0]
    p_flat = p_var.flatten(order="F")
    seen_constraints = {}

    identity_word_index = _word_to_p_cg_index((), oa, ob, ma, mb)
    if identity_word_index != 0:
        raise ValueError("Internal error: Identity word mapping failed.")

    constraints.append(gamma[0, 0] == p_var[0, 0])

    seen_constraints[()] = (0, 0)

    for i in range(dim):
        for j in range(i, dim):
            if i == 0 and j == 0:
                continue
            word_i = words[i]
            word_j = words[j]
            word_i_conj = tuple(reversed(word_i))
            combined_word = _reduce(word_i_conj + word_j)

            if not combined_word:
                constraints.append(gamma[i, j] == 0)
                continue

            constraint_key = combined_word
            if constraint_key in seen_constraints:
                prev_i, prev_j = seen_constraints[constraint_key]
                constraints.append(gamma[i, j] == gamma[prev_i, prev_j])
                continue

            p_cg_index = _word_to_p_cg_index(combined_word, oa, ob, ma, mb)
            if p_cg_index is not None:
                constraints.append(gamma[i, j] == p_flat[p_cg_index])
                seen_constraints[constraint_key] = (i, j)
            else:
                seen_constraints[constraint_key] = (i, j)

    return constraints
