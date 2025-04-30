"""Conversions between Bell inequality notations."""
import numpy as np


def cg_to_fc(cg_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behaviour from Collins-Gisin (CG) to Full Correlator (FC) notation.

    The Collins-Gisin (CG) notation for a Bell functional or behaviour is represented by a matrix:

    .. math::
        \text{CG} =
        \begin{pmatrix}
            K      & p_B(0|1) & p_B(0|2) & \dots \\
            p_A(0|1) & p(00|11) & p(00|12) & \dots \\
            p_A(0|2) & p(00|21) & p(00|22) & \dots \\
            \vdots   & \vdots   & \vdots   & \ddots
        \end{pmatrix}

    The Full Correlator (FC) notation is represented by:

    .. math::
        \text{FC} =
        \begin{pmatrix}
            K      & \langle B_1 \rangle & \langle B_2 \rangle & \dots \\
            \langle A_1 \rangle & \langle A_1 B_1 \rangle & \langle A_1 B_2 \rangle & \dots \\
            \langle A_2 \rangle & \langle A_2 B_1 \rangle & \langle A_2 B_2 \rangle & \dots \\
            \vdots   & \vdots      & \vdots      & \ddots
        \end{pmatrix}

    This function converts between these two notations.

    Examples
    ==========

    Consider the CHSH inequality in CG notation for a functional:

    .. math::
        \text{CHSH}_{CG} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1 & -1 \\
            0 & -1 & 1
        \end{pmatrix}

    Converting to FC notation:

    >>> import numpy as np
    >>> from toqito.helper import cg_to_fc
    >>> chsh_cg = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]])
    >>> cg_to_fc(chsh_cg)
    array([[ 0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.25, -0.25],
           [ 0.  , -0.25,  0.25]])

    Consider a behaviour (probability distribution) in CG notation:

    .. math::
        P_{CG} =
        \begin{pmatrix}
            1 & 0.5 & 0.5 \\
            0.5 & 0.25 & 0.25 \\
            0.5 & 0.25 & 0.25
        \end{pmatrix}

    Converting to FC notation:

    >>> import numpy as np
    >>> from toqito.helper import cg_to_fc
    >>> p_cg = np.array([[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    >>> cg_to_fc(p_cg, behaviour=True)
    array([[1., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    :param cg_mat: The matrix in Collins-Gisin notation.
    :param behaviour: If True, assume input is a behaviour (default: False, assume functional).
    :return: The matrix in Full Correlator notation.

    .. note::
        This function is adapted from the QETLAB MATLAB package function `CG2FC`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    ia = cg_mat.shape[0] - 1
    ib = cg_mat.shape[1] - 1


    fc_mat = np.zeros((ia + 1, ib + 1))


    a_vec = cg_mat[1:, 0]
    b_vec = cg_mat[0, 1:]
    c_mat = cg_mat[1:, 1:]


    if not behaviour:
        fc_mat[0, 0] = cg_mat[0, 0] + np.sum(a_vec) / 2 + np.sum(b_vec) / 2 + np.sum(c_mat) / 4
        fc_mat[1:, 0] = a_vec / 2 + np.sum(c_mat, axis=1) / 4
        fc_mat[0, 1:] = b_vec / 2 + np.sum(c_mat, axis=0) / 4
        fc_mat[1:, 1:] = c_mat / 4
    else:
        fc_mat[0, 0] = 1
        fc_mat[1:, 0] = 2 * a_vec - 1
        fc_mat[0, 1:] = 2 * b_vec - 1
        fc_mat[1:, 1:] = (
            np.ones((ia, ib))
            - 2 * a_vec[:, np.newaxis]
            - 2 * b_vec[np.newaxis, :]
            + 4 * c_mat
        )


    return fc_mat

def fc_to_cg(fc_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behaviour from Full Correlator (FC) to Collins-Gisin (CG) notation.

    The Full Correlator (FC) notation is represented by:

    .. math::
        \text{FC} =
        \begin{pmatrix}
            K      & \langle B_1 \rangle & \langle B_2 \rangle & \dots \\
            \langle A_1 \rangle & \langle A_1 B_1 \rangle & \langle A_1 B_2 \rangle & \dots \\
            \langle A_2 \rangle & \langle A_2 B_1 \rangle & \langle A_2 B_2 \rangle & \dots \\
            \vdots   & \vdots      & \vdots      & \ddots
        \end{pmatrix}

    The Collins-Gisin (CG) notation for a Bell functional or behaviour is represented by a matrix:

    .. math::
        \text{CG} =
        \begin{pmatrix}
            K      & p_B(0|1) & p_B(0|2) & \dots \\
            p_A(0|1) & p(00|11) & p(00|12) & \dots \\
            p_A(0|2) & p(00|21) & p(00|22) & \dots \\
            \vdots   & \vdots   & \vdots   & \ddots
        \end{pmatrix}

    This function converts between these two notations.

    Examples
    ==========

    Consider the CHSH inequality in FC notation for a functional:

    .. math::
        \text{CHSH}_{FC} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1/4 & -1/4 \\
            0 & -1/4 & 1/4
        \end{pmatrix}

    Converting to CG notation:

    >>> import numpy as np
    >>> from toqito.helper import fc_to_cg
    >>> chsh_fc = np.array([[0, 0, 0], [0, 0.25, -0.25], [0, -0.25, 0.25]])
    >>> fc_to_cg(chsh_fc)
    array([[ 0.,  0.,  0.],
           [ 0.,  1., -1.],
           [ 0., -1.,  1.]])

    Consider a behaviour (correlation matrix) in FC notation:

    .. math::
        P_{FC} =
        \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 0 & 0 \\
            0 & 0 & 0
        \end{pmatrix}

    Converting to CG notation:

    >>> import numpy as np
    >>> from toqito.helper import fc_to_cg
    >>> p_fc = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> fc_to_cg(p_fc, behaviour=True)
    array([[1.  , 0.5 , 0.5 ],
           [0.5 , 0.25, 0.25],
           [0.5 , 0.25, 0.25]])

    :param fc_mat: The matrix in Full Correlator notation.
    :param behaviour: If True, assume input is a behaviour (default: False, assume functional).
    :return: The matrix in Collins-Gisin notation.

    .. note::
        This function is adapted from the QETLAB MATLAB package function `FC2CG`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    ia = fc_mat.shape[0] - 1
    ib = fc_mat.shape[1] - 1


    cg_mat = np.zeros((ia + 1, ib + 1))


    a_vec = fc_mat[1:, 0]
    b_vec = fc_mat[0, 1:]
    c_mat = fc_mat[1:, 1:]


    if not behaviour:
        cg_mat[0, 0] = fc_mat[0, 0] + np.sum(c_mat) - np.sum(a_vec) - np.sum(b_vec)
        cg_mat[1:, 0] = 2 * a_vec - 2 * np.sum(c_mat, axis=1)
        cg_mat[0, 1:] = 2 * b_vec - 2 * np.sum(c_mat, axis=0)
        cg_mat[1:, 1:] = 4 * c_mat
    else:
        cg_mat[0, 0] = 1
        cg_mat[1:, 0] = (1 + a_vec) / 2
        cg_mat[0, 1:] = (1 + b_vec) / 2
        cg_mat[1:, 1:] = (
            np.ones((ia, ib))
            + a_vec[:, np.newaxis]
            + b_vec[np.newaxis, :]
            + c_mat
        ) / 4


    return cg_mat




def cg_to_fp(cg_mat: np.ndarray, desc: list[int], behaviour: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behaviour from Collins-Gisin (CG) to Full Probability (FP) notation.

    The Collins-Gisin (CG) notation for a Bell functional or behaviour is represented by a matrix
    (see :func:`cg_to_fc`). The Full Probability (FP) notation represents the full probability
    distribution :math:`V(a, b, x, y) = P(a, b | x, y)`, the probability of Alice getting outcome
    :math:`a` (0 to oa-1) and Bob getting outcome :math:`b` (0 to ob-1) given inputs :math:`x`
    (0 to ia-1) and :math:`y` (0 to ib-1). This is stored as a 4D numpy array with indices
    `V[a, b, x, y]`.

    This function converts from CG to FP notation.

    Examples
    ==========

    Consider the CHSH inequality functional in CG notation:

    .. math::
        \text{CHSH}_{CG} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1 & -1 \\
            0 & -1 & 1
        \end{pmatrix}

    Converting to FP notation (desc = [2, 2, 2, 2]):

    >>> import numpy as np
    >>> from toqito.helper import cg_to_fp
    >>> chsh_cg = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]])
    >>> desc = [2, 2, 2, 2] # oa, ob, ia, ib
    >>> cg_to_fp(chsh_cg, desc) # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 1., -1.],
             [-1.,  1.]],
            [[ 0.,  0.],
             [ 0.,  0.]]],
           [[[ 0.,  0.],
             [ 0.,  0.]],
            [[ 0.,  0.],
             [ 0.,  0.]]]])


    Consider a behaviour (probability distribution) in CG notation (desc = [2, 2, 2, 2]):

    .. math::
        P_{CG} =
        \begin{pmatrix}
            1 & 0.5 & 0.5 \\
            0.5 & 0.25 & 0.25 \\
            0.5 & 0.25 & 0.25
        \end{pmatrix}

    Converting to FP notation:

    >>> p_cg = np.array([[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    >>> desc = [2, 2, 2, 2]
    >>> cg_to_fp(p_cg, desc, behaviour=True) # doctest: +NORMALIZE_WHITESPACE
    array([[[[0.25, 0.25],
             [0.25, 0.25]],
            [[0.25, 0.25],
             [0.25, 0.25]]],
           [[[0.25, 0.25],
             [0.25, 0.25]],
            [[0.25, 0.25],
             [0.25, 0.25]]]])

    :param cg_mat: The matrix in Collins-Gisin notation.
    :param desc: A list [oa, ob, ia, ib] describing the number of outputs (oa, ob) and inputs (ia, ib).
    :param behaviour: If True, assume input is a behaviour (default: False, assume functional).
    :return: The probability tensor V[a, b, x, y] in Full Probability notation.

    .. note::
        This function is adapted from the QETLAB MATLAB package function `CG2FP`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    oa, ob, ia, ib = desc
    v_mat = np.zeros((oa, ob, ia, ib))


    def aindex(a: int, x: int) -> int:
        """CG matrix row index for Alice's outcome `a` (0..oa-2) and input `x` (0..ia-1). Returns 1-based index."""
        return 1 + a + x * (oa - 1)


    def bindex(b: int, y: int) -> int:
        """CG matrix col index for Bob's outcome `b` (0..ob-2) and input `y` (0..ib-1). Returns 1-based index."""
        return 1 + b + y * (ob - 1)


    if not behaviour:
        # Functional case logic
        k_term = cg_mat[0, 0] / (ia * ib) if ia > 0 and ib > 0 else 0
        for x in range(ia):
            for y in range(ib):
                # Fill V[a, b, x, y] for a < oa-1, b < ob-1
                for a in range(oa - 1):
                    a_term = cg_mat[aindex(a, x), 0] / ib if ib > 0 else 0
                    for b in range(ob - 1):
                        b_term = cg_mat[0, bindex(b, y)] / ia if ia > 0 else 0
                        v_mat[a, b, x, y] = (
                            k_term
                            + a_term
                            + b_term
                            + cg_mat[aindex(a, x), bindex(b, y)]
                        )
                # Fill V[a, ob-1, x, y] for a < oa-1 (last column for Bob)
                for a in range(oa - 1):
                    a_term = cg_mat[aindex(a, x), 0] / ib if ib > 0 else 0
                    v_mat[a, ob - 1, x, y] = k_term + a_term
                # Fill V[oa-1, b, x, y] for b < ob-1 (last row for Alice)
                for b in range(ob - 1):
                    b_term = cg_mat[0, bindex(b, y)] / ia if ia > 0 else 0
                    v_mat[oa - 1, b, x, y] = k_term + b_term
                # Fill V[oa-1, ob-1, x, y] (bottom-right corner)
                v_mat[oa - 1, ob - 1, x, y] = k_term


    else:  # behaviour == 1
        for x in range(ia):
            for y in range(ib):
                # Calculate slices for CG matrix corresponding to non-last outcomes
                # Need 1-based indices for slicing cg_mat
                start_row_a = aindex(0, x)
                end_row_a = aindex(oa - 2, x) + 1 if oa > 1 else start_row_a
                slice_a = slice(start_row_a, end_row_a)


                start_col_b = bindex(0, y)
                end_col_b = bindex(ob - 2, y) + 1 if ob > 1 else start_col_b
                slice_b = slice(start_col_b, end_col_b)


                # Get corresponding submatrix or default to zeros/scalars if outputs=1
                cg_sub_mat = cg_mat[slice_a, slice_b] if oa > 1 and ob > 1 else np.array([[]])
                cg_a_marg = cg_mat[slice_a, 0] if oa > 1 else np.array([])
                cg_b_marg = cg_mat[0, slice_b] if ob > 1 else np.array([])


                # V(0..oa-2, 0..ob-2, x, y) = p(a,b|xy)
                if oa > 1 and ob > 1:
                    v_mat[0 : oa - 1, 0 : ob - 1, x, y] = cg_sub_mat


                # V(0..oa-2, ob-1, x, y) = pA(a|x) - sum_{b'=0..ob-2} p(a,b'|xy)
                if oa > 1:
                    sum_b = np.sum(cg_sub_mat, axis=1) if ob > 1 else np.zeros(oa - 1)
                    v_mat[0 : oa - 1, ob - 1, x, y] = cg_a_marg - sum_b


                # V(oa-1, 0..ob-2, x, y) = pB(b|y) - sum_{a'=0..oa-2} p(a',b|xy)
                if ob > 1:
                    sum_a = np.sum(cg_sub_mat, axis=0) if oa > 1 else np.zeros(ob - 1)
                    v_mat[oa - 1, 0 : ob - 1, x, y] = cg_b_marg - sum_a


                # V(oa-1, ob-1, x, y) = 1 - sum pA(a|x) - sum pB(b|y) + sum p(ab|xy)
                sum_a_marg = np.sum(cg_a_marg)
                sum_b_marg = np.sum(cg_b_marg)
                sum_ab_joint = np.sum(cg_sub_mat)
                v_mat[oa - 1, ob - 1, x, y] = (
                    cg_mat[0, 0]      # Should be 1 for behaviour
                    - sum_a_marg
                    - sum_b_marg
                    + sum_ab_joint
                )

    return v_mat

def fc_to_fp(fc_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behaviour from Full Correlator (FC) to Full Probability (FP) notation.

    Assumes binary outcomes (oa=2, ob=2) corresponding to physical values +1 and -1.
    The FP tensor indices `a, b = 0, 1` correspond to outcomes `+1, -1` respectively.

    The Full Correlator (FC) notation is represented by a matrix (see :func:`fc_to_cg`).
    The Full Probability (FP) notation represents the full probability distribution
    :math:`V(a, b, x, y) = P(\text{out}_A=a', \text{out}_B=b' | x, y)`,
    where :math:`a=0 \rightarrow a'=+1`, :math:`a=1 \rightarrow a'=-1` (similarly for b),
    stored as a 4D numpy array `V[a, b, x, y]`.

    This function converts from FC to FP notation.

    Examples
    ==========

    Consider the CHSH inequality functional in FC notation:

    .. math::
        \text{CHSH}_{FC} =
        \begin{pmatrix}
            0 & 0 & 0 \\
            0 & 1/4 & -1/4 \\
            0 & -1/4 & 1/4
        \end{pmatrix}

    Converting to FP notation:

    >>> import numpy as np
    >>> from toqito.helper import fc_to_fp
    >>> chsh_fc = np.array([[0, 0, 0], [0, 0.25, -0.25], [0, -0.25, 0.25]])
    >>> fc_to_fp(chsh_fc) # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0.25, -0.25],
             [-0.25,  0.25]],
            [[-0.25,  0.25],
             [ 0.25, -0.25]]],
           [[[-0.25,  0.25],
             [ 0.25, -0.25]],
            [[ 0.25, -0.25],
             [-0.25,  0.25]]]])

    Consider a behaviour (correlation matrix) in FC notation (e.g., from PR box):
    Note: This FC matrix corresponds to the PR box *after* applying `fp_to_fc(pr_box, behaviour=True)`,
    which uses the QETLAB convention of averaging marginal correlators.

    .. math::
        P_{FC} =
        \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1/\sqrt{2} & 1/\sqrt{2} \\
            0 & 1/\sqrt{2} & -1/\sqrt{2}
        \end{pmatrix}

    Converting to FP notation:

    >>> p_fc = np.array([[1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)], [0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    >>> fc_to_fp(p_fc, behaviour=True) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[[[0.4267767, 0.4267767],
             [0.4267767, 0.0732233]],
    <BLANKLINE>
            [[0.0732233, 0.0732233],
             [0.0732233, 0.4267767]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[0.0732233, 0.0732233],
             [0.0732233, 0.4267767]],
    <BLANKLINE>
            [[0.4267767, 0.4267767],
             [0.4267767, 0.0732233]]]])

    :param fc_mat: The matrix in Full Correlator notation.
    :param behaviour: If True, assume input is a behaviour (default: False, assume functional).
    :return: The probability tensor V[a, b, x, y] in Full Probability notation (oa=2, ob=2).

    .. note::
        This function is adapted from the QETLAB MATLAB package function `FC2FP`.
        For `behaviour=True`, it applies the standard formula relating probabilities to correlators:
        :math:`P(a', b' | x, y) = (1 + a'\langle A_x \rangle + b'\langle B_y \rangle +`
        :math:`a'b'\langle A_x B_y \rangle) / 4`,
        where :math:`a', b' \in \{+1, -1\}`.
        Crucially, it uses the values :math:`\langle A_x \rangle` and :math:`\langle B_y \rangle` directly
        from the input `fc_mat`. If this input matrix was generated using a convention where these
        entries represent *averaged* marginal correlators (like the output of `fp_to_fc(..., behaviour=True)`),
        the resulting FP tensor might not represent a valid probability distribution (e.g., entries could be negative).

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    ia = fc_mat.shape[0] - 1
    ib = fc_mat.shape[1] - 1
    # Assumes oa=2, ob=2 based on FC notation structure
    oa, ob = 2, 2
    v_mat = np.zeros((oa, ob, ia, ib))

    if not behaviour:
        # Functional case logic
        k_term = fc_mat[0, 0] / (ia * ib) if ia > 0 and ib > 0 else 0
        for x in range(ia):
            ax_term = fc_mat[1 + x, 0] / ib if ib > 0 else 0
            for y in range(ib):
                by_term = fc_mat[0, 1 + y] / ia if ia > 0 else 0
                axby_term = fc_mat[1 + x, 1 + y]
                # V[0,0,x,y] = P(++,xy) coefficient
                v_mat[0, 0, x, y] = k_term + ax_term + by_term + axby_term
                # V[0,1,x,y] = P(+-,xy) coefficient
                v_mat[0, 1, x, y] = k_term + ax_term - by_term - axby_term
                # V[1,0,x,y] = P(-+,xy) coefficient
                v_mat[1, 0, x, y] = k_term - ax_term + by_term - axby_term
                # V[1,1,x,y] = P(--,xy) coefficient
                v_mat[1, 1, x, y] = k_term - ax_term - by_term + axby_term
    else:  # behaviour == 1
        for x in range(ia):
            ax_val = fc_mat[1 + x, 0]
            for y in range(ib):
                by_val = fc_mat[0, 1 + y]
                axby_val = fc_mat[1 + x, 1 + y]
                # V[0,0,x,y] = P(++,xy) = (1 + <Ax> + <By> + <AxBy>)/4
                v_mat[0, 0, x, y] = 1 + ax_val + by_val + axby_val
                # V[0,1,x,y] = P(+-,xy) = (1 + <Ax> - <By> - <AxBy>)/4
                v_mat[0, 1, x, y] = 1 + ax_val - by_val - axby_val
                # V[1,0,x,y] = P(-+,xy) = (1 - <Ax> + <By> - <AxBy>)/4
                v_mat[1, 0, x, y] = 1 - ax_val + by_val - axby_val
                # V[1,1,x,y] = P(--,xy) = (1 - <Ax> - <By> + <AxBy>)/4
                v_mat[1, 1, x, y] = 1 - ax_val - by_val + axby_val
        v_mat = v_mat / 4

    return v_mat

def fp_to_cg(v_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behaviour from Full Probability (FP) to Collins-Gisin (CG) notation.

    The Full Probability (FP) notation represents the full probability distribution
    :math:`V(a, b, x, y) = P(a, b | x, y)`, where :math:`a` (0 to oa-1), :math:`b` (0 to ob-1) are
    outcomes and :math:`x` (0 to ia-1), :math:`y` (0 to ib-1) are inputs. It's stored as a 4D
    numpy array `V[a, b, x, y]`. The Collins-Gisin (CG) notation for a Bell functional or
    behaviour is represented by a matrix (see :func:`cg_to_fc`).

    This function converts from FP to CG notation.

    Examples
    ==========

    Consider the CHSH inequality functional in FP notation:
    (Here V represents coefficients, not probabilities)

    >>> import numpy as np
    >>> from toqito.helper import fp_to_cg
    >>> chsh_fp = np.zeros((2, 2, 2, 2))
    >>> chsh_fp[0, 0, 0, 0] = 1
    >>> chsh_fp[0, 0, 0, 1] = -1
    >>> chsh_fp[0, 0, 1, 0] = -1
    >>> chsh_fp[0, 0, 1, 1] = 1
    >>> fp_to_cg(chsh_fp)
    array([[ 0.,  0.,  0.],
           [ 0.,  1., -1.],
           [ 0., -1.,  1.]])

    Consider a behaviour (probability distribution) in FP notation (standard PR box):

    >>> pr_box = np.zeros((2, 2, 2, 2))
    >>> pr_box[0, 0, 0, 0] = 0.5 # p(0,0|0,0)
    >>> pr_box[1, 1, 0, 0] = 0.5 # p(1,1|0,0)
    >>> pr_box[0, 0, 0, 1] = 0.5 # p(0,0|0,1)
    >>> pr_box[1, 1, 0, 1] = 0.5 # p(1,1|0,1)
    >>> pr_box[0, 0, 1, 0] = 0.5 # p(0,0|1,0)
    >>> pr_box[1, 1, 1, 0] = 0.5 # p(1,1|1,0)
    >>> pr_box[0, 1, 1, 1] = 0.5 # p(0,1|1,1)
    >>> pr_box[1, 0, 1, 1] = 0.5 # p(1,0|1,1)
    >>> fp_to_cg(pr_box, behaviour=True) # doctest: +NORMALIZE_WHITESPACE
    array([[1. , 0.5, 0.5],
           [0.5, 0.5, 0.5],
           [0.5, 0.5, 0. ]])

    :param v_mat: The probability tensor V[a, b, x, y] in Full Probability notation.
    :param behaviour: If True, assume input is a behaviour (default: False, assume functional).
    :return: The matrix in Collins-Gisin notation.

    .. note::
        This function is adapted from the QETLAB MATLAB package function `FP2CG`.
        For `behaviour=True`, it uses the QETLAB convention for calculating marginal probabilities,
        summing over the other party's outcomes for a *fixed* input setting of the other party
        (y=0 for Alice's marginal pA(a|x), x=0 for Bob's marginal pB(b|y)).

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    oa, ob, ia, ib = v_mat.shape

    alice_pars = max(0, ia * (oa - 1)) + 1 if oa > 0 else 0
    bob_pars = max(0, ib * (ob - 1)) + 1 if ob > 0 else 0

    if alice_pars == 0 or bob_pars == 0:

        if behaviour:
             raise ValueError("Behaviour case requires non-zero outputs (oa>0, ob>0).")
        cg_mat = np.zeros((alice_pars, bob_pars))
        return cg_mat

    cg_mat = np.zeros((alice_pars, bob_pars))

    def _cg_row_index(a: int, x: int) -> int:
        """Calculate 0-based CG matrix row index for Alice outcome `a` (0..oa-2) and input `x` (0..ia-1)."""
        return 1 + a + x * (oa - 1)

    def _cg_col_index(b: int, y: int) -> int:
        """Calculate 0-based CG matrix col index for Bob outcome `b` (0..ob-2) and input `y` (0..ib-1)."""
        return 1 + b + y * (ob - 1)

    if not behaviour:
        # Functional case logic
        cg_mat[0, 0] = np.sum(v_mat[oa - 1, ob - 1, :, :])

        if oa > 1:
            for a in range(oa - 1):
                for x in range(ia):
                    cg_mat[_cg_row_index(a, x), 0] = np.sum(
                        v_mat[a, ob - 1, x, :] - v_mat[oa - 1, ob - 1, x, :]
                    )

        if ob > 1:
            for b in range(ob - 1):
                for y in range(ib):
                    cg_mat[0, _cg_col_index(b, y)] = np.sum(
                        v_mat[oa - 1, b, :, y] - v_mat[oa - 1, ob - 1, :, y]
                    )

        if oa > 1 and ob > 1:
            for a in range(oa - 1):
                for b in range(ob - 1):
                    for x in range(ia):
                        for y in range(ib):
                            row_idx_0based = _cg_row_index(a, x)
                            col_idx_0based = _cg_col_index(b, y)
                            cg_mat[row_idx_0based, col_idx_0based] = (
                                v_mat[a, b, x, y]
                                - v_mat[a, ob - 1, x, y]
                                - v_mat[oa - 1, b, x, y]
                                + v_mat[oa - 1, ob - 1, x, y]
                            )


    else:  # behaviour == 1
        cg_mat[0, 0] = 1.0  # Set K=1 for behaviour

        if oa > 1 and ib > 0:
            for x in range(ia):
                for a in range(oa - 1):
                    target_row_0based = _cg_row_index(a, x)
                    cg_mat[target_row_0based, 0] = np.sum(v_mat[a, :, x, 0])
        elif oa > 1 and ib == 0:

             pass # Already initialized to 0

        if ob > 1 and ia > 0:
            for y in range(ib):
                for b in range(ob - 1):
                    target_col_0based = _cg_col_index(b, y)
                    cg_mat[0, target_col_0based] = np.sum(v_mat[:, b, 0, y])
        elif ob > 1 and ia == 0:
             pass

        if oa > 1 and ob > 1:
            for x in range(ia):
                for y in range(ib):
                    for a in range(oa - 1):
                        target_row_0based = _cg_row_index(a, x)
                        for b in range(ob - 1):
                            target_col_0based = _cg_col_index(b, y)
                            cg_mat[target_row_0based, target_col_0based] = v_mat[a, b, x, y]

    return cg_mat

def fp_to_fc(v_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert a Bell functional or behaviour from Full Probability (FP) to Full Correlator (FC) notation.

    Assumes binary outcomes (oa=2, ob=2). The FP tensor indices `a, b = 0, 1`
    correspond to physical outcomes `+1, -1` respectively.

    The Full Probability (FP) notation represents the full probability distribution
    :math:`V(a, b, x, y) = P(\text{out}_A=a', \text{out}_B=b' | x, y)`, where
    :math:`a=0 \rightarrow a'=+1`, :math:`a=1 \rightarrow a'=-1` (similarly for b),
    stored as a 4D numpy array `V[a, b, x, y]`.
    The Full Correlator (FC) notation is represented by a matrix (see :func:`fc_to_cg`).

    This function converts from FP to FC notation.

    Examples
    ==========

    Consider the CHSH inequality functional in FP notation:
    (Here V represents coefficients, not probabilities)

    >>> import numpy as np
    >>> from toqito.helper import fp_to_fc, fc_to_fp
    >>> chsh_fc = np.array([[0, 0, 0], [0, 0.25, -0.25], [0, -0.25, 0.25]])
    >>> chsh_fp = fc_to_fp(chsh_fc)
    >>> fp_to_fc(chsh_fp) # CHSH functional in FC
    array([[ 0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.25, -0.25],
           [ 0.  , -0.25,  0.25]])


    Consider a behaviour (probability distribution) in FP notation (standard PR box):

    >>> pr_box = np.zeros((2, 2, 2, 2))
    >>> pr_box[0, 0, 0, 0] = 0.5 # p(0,0|0,0)
    >>> pr_box[1, 1, 0, 0] = 0.5 # p(1,1|0,0)
    >>> pr_box[0, 0, 0, 1] = 0.5 # p(0,0|0,1)
    >>> pr_box[1, 1, 0, 1] = 0.5 # p(1,1|0,1)
    >>> pr_box[0, 0, 1, 0] = 0.5 # p(0,0|1,0)
    >>> pr_box[1, 1, 1, 0] = 0.5 # p(1,1|1,0)
    >>> pr_box[0, 1, 1, 1] = 0.5 # p(0,1|1,1)
    >>> pr_box[1, 0, 1, 1] = 0.5 # p(1,0|1,1)
    >>> fp_to_fc(pr_box, behaviour=True) # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  1., -1.]])

    :param v_mat: The probability tensor V[a, b, x, y] in Full Probability notation (oa=2, ob=2).
    :param behaviour: If True, assume input is a behaviour (default: False, assume functional).
    :return: The matrix in Full Correlator notation.

    .. note::
        This function is adapted from the QETLAB MATLAB package function `FP2FC`.
        For `behaviour=True`, it calculates the *average* marginal correlators `<Ax>` and `<By>`
        by summing over the other party's inputs and dividing by the number of inputs (ib or ia).
        The joint correlators `<AxBy>` are calculated directly for each (x, y).

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    oa, ob, ia, ib = v_mat.shape

    if oa != 2 or ob != 2:
        raise ValueError("FP to FC conversion currently only supports binary outcomes (oa=2, ob=2).")

    fc_mat = np.zeros((1 + ia, 1 + ib))

    fc_mat[0, 0] = np.sum(v_mat) # K' = sum(V), used for functional case

    for x in range(ia):
        fc_mat[x + 1, 0] = np.sum(v_mat[0, :, x, :]) - np.sum(v_mat[1, :, x, :])

    for y in range(ib):

        fc_mat[0, 1 + y] = np.sum(v_mat[:, 0, :, y]) - np.sum(v_mat[:, 1, :, y])

    # Calculate E[AxBy] for each (x,y) -> FC[x+1, y+1] component
    for x in range(ia):
        for y in range(ib):
            fc_mat[x + 1, y + 1] = (
                v_mat[0, 0, x, y] - v_mat[0, 1, x, y] - v_mat[1, 0, x, y] + v_mat[1, 1, x, y]
            )

    if not behaviour:
        fc_mat = fc_mat / 4
    else:
        fc_mat[0, 0] = 1
        if ib > 0:
            fc_mat[1:, 0] = fc_mat[1:, 0] / ib
        else:
             # If no Bob inputs, average marginal <Ax> is 0.
             fc_mat[1:, 0] = 0
        if ia > 0:
            fc_mat[0, 1:] = fc_mat[0, 1:] / ia
        else:
             # If no Alice inputs, average marginal <By> is 0.
             fc_mat[0, 1:] = 0

    return fc_mat
