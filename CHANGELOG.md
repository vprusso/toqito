# Changelog

## 0.0.5

- Fix: Bug in `swap.py`. Added test to cover bug found.

- Fix: Bug in `ppt_distinguishability.py` that prevented checking for higher
  dimensions. Added test to cover bug found.

- Fix: Bug in `ppt_distinguishability.py` that prevented checking 2-dimensional
  cases.

- Fix: Bug in `state_distinguishability.py`. Value returned was being multiplied
  by an unnecessary factor. Added other tests that would have caught this.

- Fix: Bug in `partial_trace.py`. Added test to cover bug found.

- Fix: Bug in `partial_transpose.py` for non-square matrices. Adding test cases to
  cover this previously failing case.
 
- Fix: Bug in `purity.py`. Squaring matrix was using `**2` instead of 
  `np.linalg.matrix_power` 
  
- Fix: Bug in `random_state_vector.py`. Added tests to cover.

- Fix: Bug in `schmidt_rank.py`. Added tests to cover.

- Feature: Added in `symmetric_extension_hierarchy.py`. The function is a
  hierarchy of semidefinite programs that eventually converge to the separable
  value for distinguishing an ensemble of quantum states. The first level is
  just the standard PPT distinguishability probability. Computing higher levels
  of this hierarchy can become intractable quite quickly.

- Feature: Added in ability to perform both minimum-error and unambiguous state
  discrimination in `state_distinguishability.py`. Adding additional tests to
  `test_state_distinguishability.py` to cover this extra feature.
  
- Feature: Added `majorizes.py`; a function to determine whether a vector or matrix
  majorizes another. This is used as a criterion to determine whether a quantum
  state can be converted to another via LOCC. Added `test_majorizes.py` for unit
  testing.

- Feature: Added `perfect_matchings.py`; a function that calculates all the 
  perfect matchings of a given list of objects. Added 
  `test_perfect_matchings.py` for unit testing.

- Feature: Added `breuer.py` under `states/`; a specific bound-entangled state of 
  interest. Added in `test_breuer.py` for unit testing.

- Feature: Added `brauer.py` under `states/`. Added in `test_brauer.py` for unit 
  testing.

- Feature: Added `entanglement_of_formation.py` under `state_props/`. Added in 
  `test_entanglement_of_formation.py` for unit testing.

- Feature: Added `l1_norm_coherence.py` under `state_props/`. Added 
  `test_l1_norm_coherence.py` for unit testing.

- Enhancement: Adding further tests for `symmetric_projection.py`.

- Enhancement: Adding further tests for `ppt_distinguishability.py`.

- Enhancement: Adding further tests for `reduction.py`

- Enhancement: Consolidating `conclusive_state_exclusion.py` and 
  `unambiguous_state_exclusion.py` to just `state_exclusion.py`. Adding in
  optional parameter to specify method of exclusion. Consolidated tests in
  `test_state_exclusion.py`.

## 0.0.6
  
 - Feature: Added `in_separable_ball.py` under `state_props/`. Knowing whether a
 density matrix (quantum state) is contained in the separable ball centered at
 the maximally-mixed state is useful for separability testing. Added 
 `test_in_separable_ball.py` for unit testing.

- Feature: Added in ability to perform both minimum-error and unambiguous state
  discrimination in `ppt_distinguishability.py`. Adding additional tests to
  `ppt_state_distinguishability.py` to cover this extra feature.
  
- Feature: Added in ability to compute both primal and dual optimization 
  problems in `ppt_distinguishability.py`. This gives the user the capability
  to obtain the measurement operators and also to use the computationally
  efficiency of the dual problem to make quicker numeric calculations.
 
- Feature: Added `unvec.py` under `matrix_ops/`. This feature is the inverse of
  pre-existing "vec" operation found in `vec.py`. That is, it allows one to take
  a vector and transform it to a (possibly square) matrix.
  
- Feature: Added `choi_to_kraus.py` under `channel_ops/`. This feature allows
  the user to convert a Choi matrix to a list of Kraus operators. This is the 
  inverse of the existing `kraus_to_choi.py` functionality that `toqito` 
  currently offers.
  
- Feature: Add `is_mutually_orthogonal.py` under `state_props/`. This feature
  allows the user to determine whether a given set of states (vectors) are
  mutually orthogonal with each other.
  
- Feature: Add `has_symmetric_extension.py` under `state_props/`. This feature 
  is very useful for determining whether a given state is entangled by checking
  whether there exists a symmetric extension for that state.
  
- Feature: Add `cvx_kron.py` under `helper/`. This feature allows one to compute
  the Kronecker product between two entities, where either two may be of type
  `np.ndarray` or a `cvxpy` expression.
 
- Feature: Add `is_identity.py` to `state_props/`. This feature allows one to check
  if a given matrix is equal to the identity matrix. 
 
- Enhancement: Previously ignore tests for `channel_props/` are now being run as
  part of the testing suite.
  
- Enhancement: More robust error checking and adding tolerance arguments for
  various matrix properties found in `matrix_props/`.
  
- Doc-Fix: Trailing `\`` removed in `li_norm_coherence.py`.
 
- Fix: Bug in `symmetric_extension_hierarchy.py` for `level=1` case. Added test
  to cover this bug.

- Fix: Bug in `states/tile.py` produced 3-dimensional vectors when they should
  have in fact been 9-dimensional vectors.
  
- Fix: Warning in testing suite for `test_symmetric_projection.py` for 
  previously using the deprecated `numpy.matrix` class has since been fixed and
  the warning hence resolved.
  
- Fix: The `states/isotropic.py` file no longer requires the use of the 
  deprecated `numpy.matrix` class.
  
## 0.0.7

- Feature: Added `is_unital.py` under `channel_props/`. This feature allows
  the user to determine whether a given channel (specified by either its Choi
  matrix or by its Kraus operators) is unital.
 
- Feature: Added `is_trace_preserving.py` under `channel_props/`. This feature
  allows the user to determine whether a given channel (specified by either its
  Choi matrix or by its Kraus operators) is trace-preserving.
  
- Feature: Added `log_negativity.py` under `state_props`. This feature
  allows the user to calculate the log negativty of a quantum state.
  
- Feature: Added `channel_fidelity.py` under `channel_metrics/`. This 
  feature allows the user to calculate the channel fidelity between the Choi 
  representations of two quantum channels.
  
- Feature: Added `is_idempotent.py` under `matrix_props/`. This feature
  allows the user to determine whether a given matrix is idempotent.
  
- Enhancement: Adding `rtol` and `atol` tolerance parameters for 
  `is_herm_preserving.py`.
  
- Enhancement: Improving speed of calculating the classical value of nonlocal
  game. This enhancement is taken from QETLAB which was inspired by pre-print
  arxiv:2005.13418.
  
- Enhancement: Parallel repetitions for the classical value of a nonlocal game
  is now supported.
  
- Fix: The `partial_channel.py` function has been enhanced to deal with 
  completely positive maps specified by Kraus operators as input.
  
- Fix: The GHZ state now supports either dimension or parameter `1` instead 
  of it previously being `2`

## 1.0.0

- Fix: Various documentation fixes and updates.

- Fix: Index error for unambiguous quantum state distinguishability.
 
## 1.0.1

- Fix: Permute systems had a bug where if the `inv_perm` option in 
  `permute_systems.py` was selected, the standard permutation was calculated 
  (not the inverse permutation). Further unit tests are included to catch
  similar failures.
  
- Fix: The `partial_transpose.py` function did not accurately calculate the 
  partial transpose on matrices of certain dimension. The fix for 
  `permute_systems.py` fixes the issue with `partial_transpose.py`. Further unit
  tests are included to catch similar failures.

- Fix: The `partial_trace.py` function was not accurately calculating the 
  partial trace when the argument was specified as a list of dimensions for 
  certain cases. This has been fixed and further test cases have been included 
  to prevent this from occurring.

- Fix: The `swap.py` function was not accurately swapping on all sub-systems.
  Further unit tests are included to catch similar failures.

- Fix: The `hadamard.py` function was not yielding Hadamard matrices of proper
  size and value. Fixed and added tests to cover this case.
  
- Fix: The `schmidt_decomposition.py` function was taking an incorrect argument 
  into the SVD function. Fixed and added further tests cases to cover.
  
- Fix: The `is_product_vector.py` was making use of the 
  `schmidt_decomposition.py` function incorrectly. Fixed and added further 
  test cases.

- Enhancement: Simplified code for `nonlocal_game.py` and 
  `extended_nonlocal_game.py`
  