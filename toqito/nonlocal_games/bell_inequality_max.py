"""Computes the maximum value of a Bell inequality."""
import warnings
from typing import Literal, Union

import cvxpy
import numpy as np

from toqito.helper.bell_notation_conversions import cg2fc, cg2fp, fc2fp, fp2fc
from toqito.helper.npa_hierarchy import npa_constraints


def _integer_digits(number: int, base: int, digits: int) -> np.ndarray:
    """Convert number to base representation with a fixed number of digits.

    Equivalent to MATLAB's integer_digits helper.

    :param number: The number to convert.
    :param base: The base to convert to.
    :param digits: The number of digits expected in the output.
    :return: NumPy array of digits.
    """
    dits = np.zeros(digits, dtype=int)
    temp_number = number
    for i in range(digits):
        dits[digits - 1 - i] = temp_number % base
        temp_number //= base
    if temp_number != 0:
        # This mimics MATLAB's behavior where it might truncate if 'digits' is too small,
        # but more often indicates an issue if number is too large for the digits/base.
        warnings.warn(f"Number {number} might be too large for base {base} and {digits} digits.")
    return dits


def bell_inequality_max(
    coefficients: np.ndarray,
    desc: list[int],
    notation: Literal['fp', 'fc', 'cg'],
    mtype: Literal['classical', 'quantum', 'nosignal'] = 'classical',
    k: Union[int, str] = 1,
    solver: str | None = None,
    verbose: bool = False,
    **solver_options
) -> float:
    r"""Compute the maximum value of a Bell inequality.

    This function calculates the maximum value attainable for a given Bell inequality
    under different physical assumptions: classical (Local Hidden Variable models),
    quantum (shared entanglement and local measurements), or no-signaling.

    The Bell inequality is defined by the coefficients and the scenario description.

    The implementation is based on :cite:`QETLAB_link`.

    .. math::
        \mathcal{B} = \sum_{a,b,x,y} C(a,b|x,y) P(a,b|x,y)

    Where :math:`C(a,b|x,y)` are the coefficients (specified by `coefficients` and `notation`)
    and :math:`P(a,b|x,y)` is the probability distribution (or behavior).

    For the quantum maximum (`mtype='quantum'`), this function computes an *upper bound*
    using the Navascués-Pironio-Acín (NPA) hierarchy :cite:`Navascues_2008_AConvergent`.
    The tightness of the bound depends on the level `k` of the hierarchy used.

    Examples
    ==========

    1.  Compute the classical maximum of the CHSH inequality in Full Correlator notation.

    >>> import numpy as np
    >>> from toqito.helper import bell_inequality_max
    >>> # CHSH coefficients in FC notation: E[A0B0] + E[A0B1] + E[A1B0] - E[A1B1]
    >>> chsh_fc = np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]])
    >>> desc = [2, 2, 2, 2] # oa, ob, ma, mb
    >>> classical_max = bell_inequality_max(chsh_fc, desc, 'fc', 'classical')
    >>> print(f"Classical max (CHSH): {classical_max}")
    Classical max (CHSH): 2.0

    2.  Compute the quantum upper bound (NPA level 1) for the CHSH inequality in Collins-Gisin notation.

    >>> from toqito.helper import fc2cg
    >>> chsh_cg = fc2cg(chsh_fc, behaviour=False)
    >>> quantum_bound = bell_inequality_max(chsh_cg, desc, 'cg', 'quantum', k=1)
    >>> print(f"Quantum bound (CHSH, k=1): {quantum_bound:.4f}")
    Quantum bound (CHSH, k=1): 2.8284

    3. Compute the no-signaling maximum for the CHSH inequality in Full Probability notation.

    >>> from toqito.helper import fc2fp
    >>> chsh_fp = fc2fp(chsh_fc, behaviour=False)
    >>> ns_max = bell_inequality_max(chsh_fp, desc, 'fp', 'nosignal')
    >>> print(f"No-signaling max (CHSH): {ns_max:.4f}")
    No-signaling max (CHSH): 4.0000


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param coefficients: A matrix or tensor specifying the Bell inequality coefficients.
                         The shape and interpretation depend on the `notation`.
                         - 'fp': 4D array `(oa, ob, ma, mb)` for :math:`C(a,b|x,y)`.
                         - 'fc': 2D array `(ma+1, mb+1)` for correlators (binary outcomes).
                         - 'cg': 2D array `((oa-1)*ma+1, (ob-1)*mb+1)` (Collins-Gisin).
    :param desc: A list `[oa, ob, ma, mb]` specifying the number of outputs
                 (Alice, Bob) and inputs (Alice, Bob).
    :param notation: The notation used for `coefficients`. Must be one of 'fp', 'fc', 'cg'.
    :param mtype: The type of maximum to compute: 'classical', 'quantum', or 'nosignal'.
                  Defaults to 'classical'.
    :param k: The level of the NPA hierarchy to use if `mtype='quantum'`.
              Can be an integer (e.g., 1, 2) or a string (e.g., '1+ab', '1+ab+aab').
              Defaults to 1.
    :param solver: Specify the CVXPY solver. Default is CVXPY's choice.
    :param verbose: If True, prints solver output. Default is False.
    :param solver_options: Additional keyword arguments passed to the CVXPY `problem.solve()` method.
    :return: The maximum value (or quantum upper bound) of the Bell inequality.
    :raises ValueError: If input parameters are invalid (notation, mtype, dimensions).
    :raises NotImplementedError: If classical max is requested for general outcomes with CG notation.
    :raises RuntimeError: If the optimization solver fails.

    """
    # Validate inputs
    if notation not in ['fp', 'fc', 'cg']:
        raise ValueError("Invalid notation. Choose 'fp', 'fc', or 'cg'.")
    if mtype not in ['classical', 'quantum', 'nosignal']:
        raise ValueError("Invalid mtype. Choose 'classical', 'quantum', or 'nosignal'.")
    if len(desc) != 4:
        raise ValueError("desc must be a list of length 4: [oa, ob, ma, mb].")

    oa, ob, ma, mb = desc

    # Check dimension consistency for fc/cg notation
    if notation == 'fc':
        if oa != 2 or ob != 2:
            raise ValueError("Full Correlator ('fc') notation requires binary outcomes (oa=2, ob=2).")
        if coefficients.shape != (ma + 1, mb + 1):
            raise ValueError(f"FC coefficients shape mismatch. Expected {(ma+1, mb+1)}, got {coefficients.shape}.")
    elif notation == 'cg':
        expected_shape = (1 + (oa - 1) * ma, 1 + (ob - 1) * mb)
        if coefficients.shape != expected_shape:
            raise ValueError(f"CG coefficients shape mismatch. Expected {expected_shape}, got {coefficients.shape}.")
    elif notation == 'fp':
        if coefficients.shape != (oa, ob, ma, mb):
             raise ValueError(f"FP coefficients shape mismatch. Expected {(oa, ob, ma, mb)}, got {coefficients.shape}.")



    # No-Signaling or Quantum Calculation (using Optimization)
    if mtype in ['nosignal', 'quantum']:
        # Convert coefficients to full probability notation for the objective function
        if notation == 'fp':
            coeffs_fp = coefficients
        elif notation == 'fc':
            # fc2fp requires coefficients, not behaviour
            coeffs_fp = fc2fp(coefficients, behaviour=False)
        else: # notation == 'cg'
            # cg2fp requires coefficients, not behaviour
            coeffs_fp = cg2fp(coefficients, (oa, ob), (ma, mb), behaviour=False)

        # Define the probability variables P(a,b|x,y)
        prob_vars = cvxpy.Variable((oa, ob, ma, mb), name="P(a,b|x,y)", nonneg=True)

        # Define the objective function
        # Objective: Maximize Sum_{a,b,x,y} C(a,b|x,y) * P(a,b|x,y)
        objective = cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(coeffs_fp, prob_vars)))

        # --- Define Constraints ---
        constraints = []

        # 1. Normalization Constraint: Sum_{a,b} P(a,b|x,y) = 1 for all x,y
        for x in range(ma):
            for y in range(mb):
                constraints.append(cvxpy.sum(prob_vars[:, :, x, y]) == 1)

        # 2. No-Signaling Constraints:
        # Alice's marginal P(a|x) = Sum_b P(a,b|x,y) must be independent of y
        for x in range(ma):
            for a in range(oa):
                alice_marginal_y0 = cvxpy.sum(prob_vars[a, :, x, 0])
                for y in range(1, mb):
                    constraints.append(cvxpy.sum(prob_vars[a, :, x, y]) == alice_marginal_y0)

        # Bob's marginal P(b|y) = Sum_a P(a,b|x,y) must be independent of x
        for y in range(mb):
            for b in range(ob):
                bob_marginal_x0 = cvxpy.sum(prob_vars[:, b, 0, y])
                for x in range(1, ma):
                    constraints.append(cvxpy.sum(prob_vars[:, b, x, y]) == bob_marginal_x0)

        # 3. NPA Hierarchy Constraints (only for 'quantum' type)
        if mtype == 'quantum':
            # Prepare the 'assemblage' dictionary for npa_constraints
            # The keys are (x, y) tuples, values are the P(a,b|x,y) variable slices.
            # npa_constraints assumes referee_dim=1 by default.
            assemblage = {}
            for x in range(ma):
                for y in range(mb):
                    # Ensure the variable slice matches expectation (oa x ob matrix)
                    assemblage[(x, y)] = prob_vars[:, :, x, y]

            npa_level_constraints = npa_constraints(assemblage, k=k, referee_dim=1)
            constraints.extend(npa_level_constraints)


        # --- Solve the Optimization Problem ---
        problem = cvxpy.Problem(objective, constraints)
        bmax = problem.solve(solver=solver, verbose=verbose, **solver_options)

        # Check solver status
        if problem.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
             # More specific warnings/errors based on status
            if problem.status in [cvxpy.INFEASIBLE, cvxpy.INFEASIBLE_INACCURATE]:
                raise RuntimeError(f"Optimization failed: Problem is infeasible. Status: {problem.status}")
            elif problem.status in [cvxpy.UNBOUNDED, cvxpy.UNBOUNDED_INACCURATE]:
                raise RuntimeError(f"Optimization failed: Problem is unbounded. Status: {problem.status}")
            else:
                raise RuntimeError(f"Optimization failed with status: {problem.status}")
        elif problem.status == cvxpy.OPTIMAL_INACCURATE:
             # Wrapped long warning line
             warnings.warn(
                 f"Solver finished with status: {problem.status}. Result might be inaccurate.",
                 RuntimeWarning
             )

        # If solution is valid, return it, otherwise handle potential None return from solve
        if bmax is None:
            # This might happen if solver fails badly even without raising an error in solve()
             raise RuntimeError(f"Optimization solver failed to return a value. Status: {problem.status}")
        return bmax

    # Classical Calculation (Enumerating Deterministic Strategies)
    elif mtype == 'classical':
        bmax = -np.inf

        # --- Binary Outcome Case (oa=2, ob=2) ---
        # Use faster algorithm based on Full Correlator notation
        if oa == 2 and ob == 2:
            if notation == 'fc':
                M_fc = coefficients
            elif notation == 'fp':
                # fp2fc requires behaviour=False for functional coefficients
                M_fc = fp2fc(coefficients, behaviour=False)
            else: # notation == 'cg'
                # cg2fc requires behaviour=False for functional coefficients
                M_fc = cg2fc(coefficients, behaviour=False)

            current_ma, current_mb = ma, mb
            # If Alice has fewer inputs, swap roles to iterate over fewer strategies
            if ma < mb:
                M_fc = M_fc.T
                current_ma, current_mb = mb, ma

            # Extract components from potentially transposed M_fc
            constant_term = M_fc[0, 0]
            bob_marg = M_fc[0, 1:]
            alice_marg = M_fc[1:, 0]
            correlations = M_fc[1:, 1:]

            # Iterate through all 2^mb deterministic strategies for Bob (party B')
            num_bob_strategies = 2**current_mb
            for b_idx in range(num_bob_strategies):
                b_vec = 1 - 2 * _integer_digits(b_idx, 2, current_mb)
                term1 = bob_marg @ b_vec
                term2 = np.sum(np.abs(alice_marg + correlations @ b_vec))
                temp_bmax = term1 + term2
                bmax = max(bmax, temp_bmax)

            bmax += constant_term # Add the constant term <1>

        # --- General Outcome Case ---
        # Use algorithm based on Full Probability notation
        else:
            if notation == 'fp':
                M_fp = coefficients
            elif notation == 'fc':
                 # Should be unreachable due to validation raising error earlier
                 raise ValueError("Internal logic error: FC notation reached general classical path.")
            else: # notation == 'cg'
                # The conversion from CG functional coefficients to FP functional coefficients
                # used in the original MATLAB code appears potentially incorrect or not generally applicable.
                # Calculating the classical max directly from CG coefficients is complex and not implemented.
                raise NotImplementedError(
                    "Classical maximum calculation for general outcomes (oa>2 or ob>2) "
                    "directly from Collins-Gisin ('cg') functional coefficients is not currently supported. "
                    "Please provide coefficients in 'fp' notation for this case."
                )

            # The rest of the logic assumes M_fp is available and correctly in FP format
            current_oa, current_ob = oa, ob
            current_ma, current_mb = ma, mb

            # Choose Bob as the party with the fewer deterministic strategies (oa^ma vs ob^mb)
            if oa**ma < ob**mb:
                # Swap roles: Alice becomes Bob, Bob becomes Alice
                # Permute M_fp[a,b,x,y] -> M_fp_swapped[b,a,y,x]
                M_fp = np.transpose(M_fp, (1, 0, 3, 2))
                current_oa, current_ob = ob, oa
                current_ma, current_mb = mb, ma

            # Reshape M_fp for easier calculation: M_fp[a,b,x,y] -> M_reshaped[a*x, b*y]
            # (Indices based on *current* dimensions after potential swap)
            M_reshaped = np.transpose(M_fp, (0, 2, 1, 3)) # M_fp[a, x, b, y]
            M_reshaped = M_reshaped.reshape(current_oa * current_ma, current_ob * current_mb)

            # Offset for indexing into the reshaped matrix based on Bob's strategy
            offset = np.arange(current_mb) * current_ob

            # Iterate through all ob^mb deterministic strategies for Bob (party B')
            num_bob_strategies = current_ob**current_mb
            for b_idx in range(num_bob_strategies):
                bob_choices = _integer_digits(b_idx, current_ob, current_mb)
                bob_col_indices = bob_choices + offset
                Ma = np.sum(M_reshaped[:, bob_col_indices], axis=1)
                Ma_reshaped = Ma.reshape((current_oa, current_ma))
                temp_bmax = np.sum(np.max(Ma_reshaped, axis=0))
                bmax = max(bmax, temp_bmax)

        return bmax

    else:
        # This case should not be reachable due to initial validation
        raise ValueError("Internal error: Invalid mtype reached classical calculation block.")
