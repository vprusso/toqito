"""Bell notation conversion functions."""
import numpy as np


def cg2fc(cg_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert from Collins-Gisin to Full Correlator notation :cite:Collins_2004_BellIn.

    Converts a Bell functional or behaviour from Collins-Gisin notation to Full Correlator notation.
    The Collins-Gisin notation represents:

    .. math::
        CG = \begin{pmatrix}
            K & p_B(0|1) & p_B(0|2) & \cdots \\
            p_A(0|1) & p(00|11) & p(00|12) & \cdots \\
            \vdots & \vdots & \vdots & \ddots
        \end{pmatrix}

    The Full Correlator notation represents:

    .. math::
        FC = \begin{pmatrix}
            K & \langle B_1 \rangle & \langle B_2 \rangle & \cdots \\
            \langle A_1 \rangle & \langle A_1B_1 \rangle & \langle A_1B_2 \rangle & \cdots \\
            \vdots & \vdots & \vdots & \ddots
        \end{pmatrix}

    Examples
    ==========

    Converting a Collins-Gisin matrix representing a *behaviour* to Full Correlator notation:

    >>> import numpy as np
    >>> from toqito.helper import cg2fc
    >>> # Input represents uniform distribution p(ab|xy)=1/4
    >>> cg_mat = np.array([[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    >>> # Specify behaviour=True for behaviour conversion
    >>> fc_mat = cg2fc(cg_mat, behaviour=True)
    >>> print(np.round(fc_mat, decimals=2))
    [[1. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param cg_mat: Bell functional or behaviour in Collins-Gisin notation
    :param behaviour: If True, treats input as behaviour, otherwise as Bell functional
    :return: Converted matrix in Full Correlator notation

    """
    ia = cg_mat.shape[0] - 1
    ib = cg_mat.shape[1] - 1

    fc_mat = np.zeros((ia + 1, ib + 1))

    # Extract components
    alice_marg = cg_mat[1:, 0] if ia > 0 else np.array([]) # Corresponds to A in MATLAB
    bob_marg = cg_mat[0, 1:] if ib > 0 else np.array([])   # Corresponds to B in MATLAB
    corr = cg_mat[1:, 1:] if (ia > 0 and ib > 0) else np.array([[]]).reshape(ia, ib) # Corresponds to C in MATLAB

    if not behaviour:
        k_val = cg_mat[0, 0]
        # Convert Bell functional
        fc_mat[0, 0] = k_val + np.sum(alice_marg)/2 + np.sum(bob_marg)/2 + np.sum(corr)/4
        if ia > 0:
            fc_mat[1:, 0] = alice_marg/2 + np.sum(corr, axis=1)/4
        if ib > 0:
            fc_mat[0, 1:] = bob_marg/2 + np.sum(corr, axis=0)/4
        if ia > 0 and ib > 0:
            fc_mat[1:, 1:] = corr/4
    else:
        # Convert behaviour
        fc_mat[0, 0] = 1
        if ia > 0:
            fc_mat[1:, 0] = 2*alice_marg - 1
        if ib > 0:
            fc_mat[0, 1:] = 2*bob_marg - 1
        # Replicate MATLAB's ones(ia,ib) - 2*A*ones(1,ib) - 2*ones(ia,1)*B + 4*C using broadcasting
        if ia > 0 and ib > 0:
            fc_mat[1:, 1:] = (np.ones((ia, ib)) -
                             2 * alice_marg[:, np.newaxis] - # Broadcasts A correctly
                             2 * bob_marg[np.newaxis, :] +   # Broadcasts B correctly
                             4 * corr)

    return fc_mat


def fc2cg(fc_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert from Full Correlator to Collins-Gisin notation :cite:Collins_2004_BellIn.

    Converts a Bell functional or behaviour from Full Correlator notation to Collins-Gisin notation.
    The Full Correlator notation represents:

    .. math::
        FC = \begin{pmatrix}
            K & \langle B_1 \rangle & \langle B_2 \rangle & \cdots \\
            \langle A_1 \rangle & \langle A_1B_1 \rangle & \langle A_1B_2 \rangle & \cdots \\
            \vdots & \vdots & \vdots & \ddots
        \end{pmatrix}

    The Collins-Gisin notation represents:

    .. math::
        CG = \begin{pmatrix}
            K & p_B(0|1) & p_B(0|2) & \cdots \\
            p_A(0|1) & p(00|11) & p(00|12) & \cdots \\
            \vdots & \vdots & \vdots & \ddots
        \end{pmatrix}

    Examples
    ==========

    Converting a Full Correlator matrix representing a *behaviour* to Collins-Gisin notation:

    >>> import numpy as np
    >>> from toqito.helper import fc2cg
    >>> # Input correlators correspond to uniform distribution behaviour
    >>> fc_mat = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> # Specify behaviour=True
    >>> cg_mat = fc2cg(fc_mat, behaviour=True)
    >>> print(np.round(cg_mat, decimals=2))
    [[1.   0.5  0.5 ]
     [0.5  0.25 0.25]
     [0.5  0.25 0.25]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param fc_mat: Bell functional or behaviour in Full Correlator notation
    :param behaviour: If True, treats input as behaviour, otherwise as Bell functional
    :return: Converted matrix in Collins-Gisin notation

    """
    ia = fc_mat.shape[0] - 1
    ib = fc_mat.shape[1] - 1

    cg_mat = np.zeros((ia + 1, ib + 1))

    # Extract components
    alice_corr = fc_mat[1:, 0] if ia > 0 else np.array([])   # Corresponds to A in MATLAB
    bob_corr = fc_mat[0, 1:] if ib > 0 else np.array([])     # Corresponds to B in MATLAB
    joint_corr = fc_mat[1:, 1:] if (ia > 0 and ib > 0) else np.array([[]]).reshape(ia, ib)  # Corresponds to C in MATLAB

    if not behaviour:
        k_val = fc_mat[0, 0]
        # Convert Bell functional
        cg_mat[0, 0] = k_val + np.sum(joint_corr) - np.sum(alice_corr) - np.sum(bob_corr)
        if ia > 0:
            cg_mat[1:, 0] = 2*alice_corr - 2*np.sum(joint_corr, axis=1)
        if ib > 0:
            cg_mat[0, 1:] = 2*bob_corr - 2*np.sum(joint_corr, axis=0)
        if ia > 0 and ib > 0:
            cg_mat[1:, 1:] = 4*joint_corr
    else:
        # Convert behaviour
        cg_mat[0, 0] = 1
        if ia > 0:
            cg_mat[1:, 0] = (1 + alice_corr)/2
        if ib > 0:
            cg_mat[0, 1:] = (1 + bob_corr)/2
        # Replicate MATLAB's (ones(ia,ib) + A*ones(1,ib) + ones(ia,1)*B + C)/4 using broadcasting
        if ia > 0 and ib > 0:
            cg_mat[1:, 1:] = (np.ones((ia, ib)) +
                             alice_corr[:, np.newaxis] + # Broadcasts A correctly
                             bob_corr[np.newaxis, :] +   # Broadcasts B correctly
                             joint_corr) / 4

    return cg_mat

def cg2fp(cg_mat: np.ndarray, output_dim: tuple[int, int],
          input_dim: tuple[int, int], behaviour: bool = False) -> np.ndarray:
    r"""Convert from Collins-Gisin to Full Probability notation :cite:Collins_2004_BellIn.

    Converts a Bell functional or behaviour from Collins-Gisin notation to Full Probability notation
    V(a,b,x,y) where a,b are outputs and x,y are inputs.

    Examples
    ==========

    Converting a Collins-Gisin matrix representing a *behaviour* to Full Probability notation:

    >>> import numpy as np
    >>> from toqito.helper import cg2fp
    >>> # Input represents uniform distribution p(ab|xy)=1/4
    >>> cg_mat = np.array([[1, 0.5], [0.5, 0.25]])
    >>> # Specify behaviour=True
    >>> fp_mat = cg2fp(cg_mat, (2,2), (1,1), behaviour=True)
    >>> print(np.round(fp_mat, decimals=2))
    [[[[0.25]]
    <BLANKLINE>
      [[0.25]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[0.25]]
    <BLANKLINE>
      [[0.25]]]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param cg_mat: Bell functional or behaviour in Collins-Gisin notation
    :param output_dim: Tuple (oa,ob) with number of outputs for Alice and Bob
    :param input_dim: Tuple (ia,ib) with number of inputs for Alice and Bob
    :param behaviour: If True, treats input as behaviour, otherwise as Bell functional
    :return: 4D array in Full Probability notation V[a,b,x,y]

    """
    oa, ob = output_dim
    ia, ib = input_dim

    # Helper function definitions
    def get_a_row_idx(a, x): # a=0..oa-2, x=0..ia-1
        if oa <= 1:
            # Fixed E701: Moved statement to next line
            raise IndexError("a index requested but oa=1")
        # Added check for ia > 0, although main logic should prevent call if ia=0
        if ia <= 0:
            # Fixed E701: Moved statement to next line
            raise IndexError("x index requested but ia=0")
        return 1 + a + x * (oa - 1)

    def get_b_col_idx(b, y): # b=0..ob-2, y=0..ib-1
        if ob <= 1:
            # Fixed E701: Moved statement to next line
            raise IndexError("b index requested but ob=1")
         # Added check for ib > 0, although main logic should prevent call if ib=0
        if ib <= 0:
            # Fixed E701: Moved statement to next line
            raise IndexError("y index requested but ib=0")
        return 1 + b + y * (ob - 1)

    v_mat = np.zeros((oa, ob, ia, ib))

    if not behaviour:
        # Convert Bell functional
        k_term = cg_mat[0, 0] / (ia * ib) if (ia > 0 and ib > 0) else 0 # Added check for ia/ib=0

        for x in range(ia):
            for y in range(ib):
                # Handle a=0..oa-2, b=0..ob-2 case (corresponds to MATLAB a=1..oa-1, b=1..ob-1)
                if oa > 1 and ob > 1: # Check if indices are valid
                    for a in range(oa-1):
                        for b in range(ob-1):
                            a_row = get_a_row_idx(a, x)
                            b_col = get_b_col_idx(b, y)
                            # Added checks for ia/ib > 0 before division
                            term_a = cg_mat[a_row, 0] / ib if ib > 0 else 0
                            term_b = cg_mat[0, b_col] / ia if ia > 0 else 0
                            v_mat[a,b,x,y] = (k_term + term_a + term_b + cg_mat[a_row, b_col])

                # Handle a=0..oa-2, b=ob-1 case (corresponds to MATLAB a=1..oa-1, b=ob)
                if oa > 1: # Check if index 'a' is valid
                    for a in range(oa-1):
                        a_row = get_a_row_idx(a, x)
                        # Check if ib > 0 to avoid division by zero
                        term_a = cg_mat[a_row, 0] / ib if ib > 0 else 0
                        v_mat[a, ob-1, x, y] = k_term + term_a

                # Handle a=oa-1, b=0..ob-2 case (corresponds to MATLAB a=oa, b=1..ob-1)
                if ob > 1: # Check if index 'b' is valid
                    for b in range(ob-1):
                        b_col = get_b_col_idx(b, y)
                        # Check if ia > 0 to avoid division by zero
                        term_b = cg_mat[0, b_col] / ia if ia > 0 else 0
                        v_mat[oa-1, b, x, y] = k_term + term_b

                # Handle a=oa-1, b=ob-1 case (corresponds to MATLAB a=oa, b=ob)
                v_mat[oa-1, ob-1, x, y] = k_term

    else:
        # Convert behaviour
        for x in range(ia):
            for y in range(ib):
                # --- Revised logic for behaviour=True ---
                if oa == 1 and ob == 1:
                    # Check shape integrity
                    if cg_mat.shape == (1, 1):
                        v_mat[0, 0, x, y] = cg_mat[0, 0]
                    else:
                        # Fixed E501: Broke long line
                        msg = (f"Expected cg_mat shape (1,1) for oa=1, ob=1,"
                               f" ia={ia}, ib={ib}, got {cg_mat.shape}")
                        raise ValueError(msg)
                elif oa == 1 and ob > 1:
                    # Check shape integrity
                    expected_cols = 1 + ib * (ob - 1)
                    if cg_mat.shape == (1, expected_cols):
                        # Get Bob's marginals P(b|y) for b=0..ob-2
                        bob_marg_block = np.zeros(ob - 1)
                        if ib > 0: # Need ib > 0 to get indices
                            for b_idx in range(ob - 1):
                                b_col = get_b_col_idx(b_idx, y)
                                bob_marg_block[b_idx] = cg_mat[0, b_col]
                        # Assign probabilities P(0, b|x,y) = P(b|y)
                        v_mat[0, 0:ob-1, x, y] = bob_marg_block
                        # Calculate P(0, ob-1 | x, y) = 1 - sum_{b=0..ob-2} P(b|y)
                        v_mat[0, ob-1, x, y] = cg_mat[0, 0] - np.sum(bob_marg_block)
                    else:
                        # Fixed E501: Broke long line
                        msg = (f"Expected cg_mat shape (1,{expected_cols}) for oa=1,"
                               f" ob={ob}, ia={ia}, ib={ib}, got {cg_mat.shape}")
                        raise ValueError(msg)

                elif oa > 1 and ob == 1:
                     # Check shape integrity
                    expected_rows = 1 + ia * (oa - 1)
                    if cg_mat.shape == (expected_rows, 1):
                        # Get Alice's marginals P(a|x) for a=0..oa-2
                        alice_marg_block = np.zeros(oa - 1)
                        if ia > 0: # Need ia > 0 to get indices
                             for a_idx in range(oa - 1):
                                a_row = get_a_row_idx(a_idx, x)
                                alice_marg_block[a_idx] = cg_mat[a_row, 0]
                        # Assign probabilities P(a, 0 | x,y) = P(a|x)
                        v_mat[0:oa-1, 0, x, y] = alice_marg_block
                        # Calculate P(oa-1, 0 | x, y) = 1 - sum_{a=0..oa-2} P(a|x)
                        v_mat[oa-1, 0, x, y] = cg_mat[0, 0] - np.sum(alice_marg_block)
                    else:
                        # Fixed E501: Broke long line
                        msg = (f"Expected cg_mat shape ({expected_rows},1) for oa={oa},"
                               f" ob=1, ia={ia}, ib={ib}, got {cg_mat.shape}")
                        raise ValueError(msg)

                else: # oa > 1 and ob > 1 (General case)
                    # Check shape integrity
                    expected_rows = 1 + ia * (oa - 1)
                    expected_cols = 1 + ib * (ob - 1)
                    if cg_mat.shape == (expected_rows, expected_cols):
                        # --- Logic for general case (as before) ---
                        # Get indices for blocks
                        a_start_row = get_a_row_idx(0, x)
                        a_end_row = get_a_row_idx(oa-2, x) + 1
                        b_start_col = get_b_col_idx(0, y)
                        b_end_col = get_b_col_idx(ob-2, y) + 1

                        # Extract the main block P(a,b|xy) for a<oa-1, b<ob-1
                        prob_block = cg_mat[a_start_row:a_end_row, b_start_col:b_end_col]
                        # Extract marginals P(a|x) for a<oa-1 and P(b|y) for b<ob-1
                        alice_marg_block = cg_mat[a_start_row:a_end_row, 0]
                        bob_marg_block = cg_mat[0, b_start_col:b_end_col]

                        # Assign main block
                        v_mat[0:oa-1, 0:ob-1, x, y] = prob_block
                        # Calculate P(a, ob-1 | x, y) = P(a|x) - sum_{b<ob-1} P(a,b|x,y)
                        v_mat[0:oa-1, ob-1, x, y] = alice_marg_block - np.sum(prob_block, axis=1)
                        # Calculate P(oa-1, b | x, y) = P(b|y) - sum_{a<oa-1} P(a,b|x,y)
                        v_mat[oa-1, 0:ob-1, x, y] = bob_marg_block - np.sum(prob_block, axis=0)
                        # Calculate P(oa-1, ob-1 | x, y)
                        v_mat[oa-1, ob-1, x, y] = (cg_mat[0,0] - np.sum(alice_marg_block)
                                                  - np.sum(bob_marg_block)
                                                  + np.sum(prob_block))
                    else:
                        # Fixed E501: Broke long line
                        msg = (f"Expected cg_mat shape ({expected_rows},{expected_cols})"
                               f" for oa={oa}, ob={ob}, ia={ia}, ib={ib},"
                               f" got {cg_mat.shape}")
                        raise ValueError(msg)
    return v_mat


def fp2cg(fp_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert from Full Probability to Collins-Gisin notation :cite:Collins_2004_BellIn.

    Converts a Bell functional or behaviour from Full Probability notation V(a,b,x,y)
    to Collins-Gisin notation.

    Examples
    ==========

    Converting a Full Probability matrix to Collins-Gisin notation:

    >>> import numpy as np
    >>> from toqito.helper import fp2cg
    >>> fp_mat = np.ones((2, 2, 1, 1)) * 0.25 # Uniform distribution
    >>> cg_mat = fp2cg(fp_mat, True)
    >>> print(np.round(cg_mat, decimals=2))
    [[1.   0.5 ]
     [0.5  0.25]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param fp_mat: 4D array in Full Probability notation V[a,b,x,y]
    :param behaviour: If True, treats input as behaviour, otherwise as Bell functional
    :return: Converted matrix in Collins-Gisin notation

    """
    oa, ob, ia, ib = fp_mat.shape
    cg_rows = max(1, 1 + ia*(oa-1)) if oa > 1 else 1
    cg_cols = max(1, 1 + ib*(ob-1)) if ob > 1 else 1
    cg_mat = np.zeros((cg_rows, cg_cols))

    # Helper function definitions
    def get_a_row_idx(a, x): # a=0..oa-2, x=0..ia-1
        if oa <= 1:
            # Fixed E701: Moved statement to next line
            return 0
        if ia <= 0:
            # Fixed E701: Moved statement to next line
            raise IndexError("x index requested but ia=0")
        return 1 + a + x * (oa - 1)

    def get_b_col_idx(b, y): # b=0..ob-2, y=0..ib-1
        if ob <= 1:
            # Fixed E701: Moved statement to next line
            return 0
        if ib <= 0:
            # Fixed E701: Moved statement to next line
            raise IndexError("y index requested but ib=0")
        return 1 + b + y * (ob - 1)

    if not behaviour:
        # Convert Bell functional
        cg_mat[0,0] = np.sum(fp_mat[oa-1, ob-1, :, :]) # Sum over x, y

        # Convert Alice's marginals component (only if oa > 1)
        if oa > 1:
            # Check ia > 0 before looping over x
            if ia > 0:
                for a in range(oa-1): # a = 0..oa-2
                    for x in range(ia): # x = 0..ia-1
                        a_row = get_a_row_idx(a, x)
                        # Added check for ib > 0
                        sum_term = np.sum(fp_mat[a, ob-1, x, :] - fp_mat[oa-1, ob-1, x, :]) if ib > 0 else 0
                        cg_mat[a_row, 0] = sum_term

        # Convert Bob's marginals component (only if ob > 1)
        if ob > 1:
             # Check ib > 0 before looping over y
            if ib > 0:
                for b in range(ob-1): # b = 0..ob-2
                    for y in range(ib): # y = 0..ib-1
                        b_col = get_b_col_idx(b, y)
                         # Added check for ia > 0
                        sum_term = np.sum(fp_mat[oa-1, b, :, y] - fp_mat[oa-1, ob-1, :, y]) if ia > 0 else 0
                        cg_mat[0, b_col] = sum_term

        # Convert correlations component (only if oa > 1 and ob > 1)
        if oa > 1 and ob > 1:
            # Check ia > 0 and ib > 0 before looping
            if ia > 0 and ib > 0:
                for a in range(oa-1):
                    for b in range(ob-1):
                        for x in range(ia):
                            for y in range(ib):
                                a_row = get_a_row_idx(a, x)
                                b_col = get_b_col_idx(b, y)
                                cg_mat[a_row, b_col] = (
                                    fp_mat[a, b, x, y] -
                                    fp_mat[a, ob-1, x, y] -
                                    fp_mat[oa-1, b, x, y] +
                                    fp_mat[oa-1, ob-1, x, y])
    else:
        # Convert behaviour
        cg_mat[0,0] = 1

        # Convert Alice's marginals pA(a|x) = sum_b V(a,b,x,y=0) (only if oa > 1)
        if oa > 1:
            # Check ia > 0 before looping over x
            if ia > 0:
                for a in range(oa-1): # a = 0..oa-2
                    for x in range(ia): # x = 0..ia-1
                        a_row = get_a_row_idx(a, x)
                        # Sum over b for the first y setting (index 0)
                        # Added check for ib > 0
                        # Ensure fp_mat has the y dimension before summing over it
                        cg_mat[a_row, 0] = np.sum(fp_mat[a, :, x, 0]) if ib > 0 else 0

        # Convert Bob's marginals pB(b|y) = sum_a V(a,b,x=0,y) (only if ob > 1)
        if ob > 1:
            # Check ib > 0 before looping over y
            if ib > 0:
                for b in range(ob-1): # b = 0..ob-2
                    for y in range(ib): # y = 0..ib-1
                        b_col = get_b_col_idx(b, y)
                        # Sum over a for the first x setting (index 0)
                        # Added check for ia > 0
                        # Ensure fp_mat has the x dimension before summing over it
                        cg_mat[0, b_col] = np.sum(fp_mat[:, b, 0, y]) if ia > 0 else 0

        # Convert correlations p(ab|xy) for non-last outcomes (only if oa > 1 and ob > 1)
        if oa > 1 and ob > 1:
             # Check ia > 0 and ib > 0 before looping
            if ia > 0 and ib > 0:
                for x in range(ia):
                    for y in range(ib):
                        # Get index ranges corresponding to MATLAB's aindex/bindex for the block
                        a_start_row = get_a_row_idx(0, x)
                        a_end_row = get_a_row_idx(oa-2, x) + 1 # +1 for Python slicing endpoint
                        b_start_col = get_b_col_idx(0, y)
                        b_end_col = get_b_col_idx(ob-2, y) + 1 # +1 for Python slicing endpoint
                        # Ensure indices are within bounds before slicing
                        # Check that rows/cols exist before assigning
                        if a_start_row <= a_end_row and b_start_col <= b_end_col and \
                           a_end_row <= cg_mat.shape[0] and b_end_col <= cg_mat.shape[1]:
                            cg_mat[a_start_row:a_end_row, b_start_col:b_end_col] = fp_mat[0:oa-1, 0:ob-1, x, y]

    return cg_mat


def fc2fp(fc_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert from Full Correlator to Full Probability notation :cite:Collins_2004_BellIn.

    Converts a Bell functional or behaviour from Full Correlator notation to full
    probability notation V(a,b,x,y). Only works for binary outcomes (oa=2, ob=2).

    Examples
    ==========

    Converting a Full Correlator matrix to Full Probability notation:

    >>> import numpy as np
    >>> from toqito.helper import fc2fp
    >>> fc_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # ia=2, ib=2
    >>> # Example for functional (behaviour=False)
    >>> fp_mat_func = fc2fp(fc_mat, False)
    >>> print(np.round(fp_mat_func, decimals=2))
    [[[[ 1.25  0.25]
       [ 0.25  1.25]]
    <BLANKLINE>
      [[-0.75  0.25]
       [ 0.25 -0.75]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[-0.75  0.25]
       [ 0.25 -0.75]]
    <BLANKLINE>
      [[ 1.25  0.25]
       [ 0.25  1.25]]]]
    >>> # Example for behaviour (behaviour=True)
    >>> fp_mat_beh = fc2fp(fc_mat, True)
    >>> print(np.round(fp_mat_beh, decimals=2))
    [[[[0.5  0.25]
       [0.25 0.5 ]]
    <BLANKLINE>
      [[0.   0.25]
       [0.25 0.  ]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[0.   0.25]
       [0.25 0.  ]]
    <BLANKLINE>
      [[0.5  0.25]
       [0.25 0.5 ]]]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param fc_mat: Bell functional or behaviour in Full Correlator notation
    :param behaviour: If True, treats input as behaviour, otherwise as Bell functional
    :return: 4D array in Full Probability notation V[a,b,x,y]
    :raises ValueError: If fc_mat does not imply binary outcomes (shape requires at least 1 input for A and B)

    """
    ia = fc_mat.shape[0] - 1
    ib = fc_mat.shape[1] - 1

    if ia < 0 or ib < 0:
        raise ValueError("Input fc_mat shape must be at least (1, 1).")

    # Initialize output array (2x2 outcomes only)
    v_mat = np.zeros((2, 2, ia, ib))

    k_val = fc_mat[0,0]
    alice_corr = fc_mat[1:,0] if ia > 0 else np.array([]) # <A_x> for x=0..ia-1
    bob_corr = fc_mat[0,1:] if ib > 0 else np.array([])  # <B_y> for y=0..ib-1
    joint_corr = fc_mat[1:,1:] if (ia > 0 and ib > 0) else np.array([[]]).reshape(ia, ib) # <A_x B_y>

    if not behaviour:
        # Convert Bell functional
        # Avoid division by zero if ia or ib is 0
        k_term = k_val / (ia * ib) if (ia > 0 and ib > 0) else 0

        for x in range(ia):
            for y in range(ib):
                ax = alice_corr[x]
                by = bob_corr[y]
                cxy = joint_corr[x,y]
                # Added checks for ia/ib > 0 before division
                ax_term = ax / ib if ib > 0 else 0
                by_term = by / ia if ia > 0 else 0
                # Indices a,b correspond to MATLAB 1,2
                # V(1,1,x,y) -> v_mat[0,0,x,y]
                v_mat[0,0,x,y] = k_term + ax_term + by_term + cxy
                # V(1,2,x,y) -> v_mat[0,1,x,y]
                v_mat[0,1,x,y] = k_term + ax_term - by_term - cxy
                # V(2,1,x,y) -> v_mat[1,0,x,y]
                v_mat[1,0,x,y] = k_term - ax_term + by_term - cxy
                # V(2,2,x,y) -> v_mat[1,1,x,y]
                v_mat[1,1,x,y] = k_term - ax_term - by_term + cxy
    else:
        # Convert behaviour
        for x in range(ia):
            for y in range(ib):
                ax = alice_corr[x] if ia > 0 else 0 # Default to 0 if no Alice inputs
                by = bob_corr[y] if ib > 0 else 0 # Default to 0 if no Bob inputs
                cxy = joint_corr[x,y] if (ia > 0 and ib > 0) else 0 # Default to 0 if no joint inputs
                # Indices a,b correspond to MATLAB 1,2
                # V(1,1,x,y) -> v_mat[0,0,x,y]
                v_mat[0,0,x,y] = 1 + ax + by + cxy
                # V(1,2,x,y) -> v_mat[0,1,x,y]
                v_mat[0,1,x,y] = 1 + ax - by - cxy
                # V(2,1,x,y) -> v_mat[1,0,x,y]
                v_mat[1,0,x,y] = 1 - ax + by - cxy
                # V(2,2,x,y) -> v_mat[1,1,x,y]
                v_mat[1,1,x,y] = 1 - ax - by + cxy
        v_mat = v_mat/4

    return v_mat


def fp2fc(fp_mat: np.ndarray, behaviour: bool = False) -> np.ndarray:
    r"""Convert from Full Probability to Full Correlator notation :cite:Collins_2004_BellIn.

    Converts a Bell functional or behaviour from Full Probability notation V(a,b,x,y)
    to Full Correlator notation. Only works for binary outcomes (oa=2, ob=2).

    Examples
    ==========

    Converting a Full Probability matrix to Full Correlator notation:

    >>> import numpy as np
    >>> from toqito.helper import fp2fc
    >>> fp_mat = np.ones((2, 2, 1, 1)) * 0.25 # Uniform distribution
    >>> fc_mat = fp2fc(fp_mat, True)
    >>> print(np.round(fc_mat, decimals=2))
    [[1. 0.]
     [0. 0.]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param fp_mat: 4D array in Full Probability notation V[a,b,x,y]
    :param behaviour: If True, treats input as behaviour, otherwise as Bell functional
    :return: Converted matrix in Full Correlator notation
    :raises ValueError: If input dimensions are not for binary outcomes (oa=2, ob=2)

    """
    oa, ob, ia, ib = fp_mat.shape
    if oa != 2 or ob != 2:
       raise ValueError("fp2fc only works with binary outcomes (oa=2, ob=2)")

    # Ensure dimensions are at least 1x1
    fc_rows = max(1, 1 + ia)
    fc_cols = max(1, 1 + ib)
    fc_mat = np.zeros((fc_rows, fc_cols))

    # Calculate K term (sum over all probabilities)
    # Sum over empty axes results in 0, which is correct.
    fc_mat[0,0] = np.sum(fp_mat)

    # Calculate Alice correlators sum (only if ia > 0)
    if ia > 0:
        for x in range(ia):
            # Sum over b and y. Sum over empty y axis is 0.
            fc_mat[x+1, 0] = np.sum(fp_mat[0, :, x, :] - fp_mat[1, :, x, :])

    # Calculate Bob correlators sum (only if ib > 0)
    if ib > 0:
        for y in range(ib):
            # Sum over a and x. Sum over empty x axis is 0.
            fc_mat[0, y+1] = np.sum(fp_mat[:, 0, :, y] - fp_mat[:, 1, :, y])

    # Calculate Joint correlators (only if ia > 0 and ib > 0)
    if ia > 0 and ib > 0:
        for x in range(ia):
            for y in range(ib):
                fc_mat[x+1, y+1] = (fp_mat[0, 0, x, y] + fp_mat[1, 1, x, y] -
                                  fp_mat[0, 1, x, y] - fp_mat[1, 0, x, y])

    if behaviour:
        # For behaviour: K=1, <A_x> = sum / ib, <B_y> = sum / ia
        fc_mat[0,0] = 1
        if ia > 0: # Check ia before accessing rows 1:
             # Avoid division by zero -> Inf/NaN or error if ib is 0
            if ib > 0:
                fc_mat[1:, 0] = fc_mat[1:, 0] / ib
            else: # If ib=0, correlators involving Bob must be 0
                 fc_mat[1:, 0] = 0
        if ib > 0: # Check ib before accessing cols 1:
            # Avoid division by zero -> Inf/NaN or error if ia is 0
            if ia > 0:
                fc_mat[0, 1:] = fc_mat[0, 1:] / ia
            else: # If ia=0, correlators involving Alice must be 0
                 fc_mat[0, 1:] = 0
    else:
        # For functional: Divide everything by 4
        fc_mat = fc_mat / 4

    return fc_mat
