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

- Doc-Fix: Trailing back-tick quotes removed in `li_norm_coherence.py`.
 
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

- Feature: Added `bures_distance.py` under `state_metrics/`. This feature
  allows the user to determine whether a given matrix is idempotent.

- Feature: Added `singlet.py` under `states/`. This feature allows one to yield
  a singlet state of dimension `dim`.

- Feature: Added `is_quantum_channel` under `channel_props`. This feature
  allows one to check whether a given input provided as either a list of Kraus
  operators, or, a Choi matrix constitutes a valid quantum channel.

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

- Enhancement: Adding ability for `schmidt_rank.py` function to process not 
  just vectors, but also matrices. Adding in unit tests to cover this case.

- Enhancement: Adding the ability for `is_product.py` function to process not
  just vectors, but also matrices. Adding in unit tests to cover this case.

- Enhancement: Simplified code for `nonlocal_game.py` and 
  `extended_nonlocal_game.py`

- Enhancement: Some general documentation clean-up. 

## 1.0.2

- Enhancement: Update to Python 3.9 from 3.7.

- Feature: Added `dual_channel.py` under `channel_ops`. This feature
  allows one to calculate the dual channel of a given input provided as either a 
  list of Kraus operators, or, a Choi matrix that represents a valid quantum 
  channel.
  
- Fix: Thanks to Jake Xuereb for a fix in the `concurrence.py` function that was
  incorrectly not taking the transpose.

- Feature: Thanks to Joseph Li (@jli0108) for the ability to calculate the Schmidt
  decomposition.  

- Feature: Added `diamond_norm.py` under `channel_metrics/`. This 
  feature allows the user to calculate the diamond norm between the Choi 
  representations of two quantum channels.

- Feature: Added `is_npt.py` under `state_props/`. This 
  feature allows the user to determine whether a given state is has negative
  partial transpose (is NPT).

- Fix: Thanks to @iftachyakar for the state distinguishability docs update

- Fix: Thanks to @georgios-ts for the fix to `symmetric_extension_hierarchy`.

- Feature: Thanks to @georgios-ts for the feature of the `sk_vector_norm`.

- Feature: Thanks to @georgios-ts for the extended nonlocal game version of the
  NPA hierarchy in `extended_nonlocal_games`.

- Fix: Thanks to @georgios-ts for a fix in the `partial_transpose.py` function.

- Feature: Thanks to @georgios-ts for the `block_positive.py` feature that
  allows the user to detect if a matrix is block positive.

- Feature: Thanks to @meandmytram for the `is_separable.py` feature that allows
  the user to test whether a quantum state represented as a density operator is
  entangled or separable.
  
- Feature: Added `standard_basis.py` under `matrices/` that allows one to 
  construct the standard basis of dimension "d".

- Feature: Added `is_orthonormal.py` under `matrix_props/` that allows one to 
  determine if a collection of vectors or matrices are orthonormal.

- Feature: Thanks to @AdithyaSireesh for `hilbert_schmidt_inner_product.py`
  under `state_metrics/` that allows one to calculate the Hilbert-Schmidt inner
  product between two linear operators.

- Enhancement: Updating `Union` syntax using the `|` operator instead of `Union`
  from `typing`.

- Enhancement: Linting and stylistic improvements.

## 1.0.3

- Version bump to fix Python package versioning issue.

## 1.0.4

- Version bump to fix Python package versioning issue.

- Adding `outer_product` and `inner_product` functions in `matrix_ops` (thanks
  to @juliusw352)

- Fixing random number generation in `random_density_matrix` function to be
  obtained from Haar measure (thanks to @BCasale)

- Adding `to_nonlocal_game` function in `XORGame` that converts an XOR game
  definition to a `NonlocalGame` object (thanks to @juliusw352)

- Adding `nonsignaling` function in `XORGame` that calculates the nonsignaling
  value of an `XORGame` object (thanks to @juliusw352)

## 1.0.5

- Adding ability to perform parallel repetition for `XORGame` objects (thanks to
  @juliusw352) [PR-143](https://github.com/vprusso/toqito/pull/143)

- Thanks to @puva-thakre for updating the installation and contribution
  guidelines. [PR-153](https://github.com/vprusso/toqito/pull/153)

- Thanks to @ryanprior for providing a CI/CD testing and linting pipeline
  [PR-154](https://github.com/vprusso/toqito/pull/154)

- Thanks to @georgios-ts for fixing a bug in `apply_channel`
  [PR-155](https://github.com/vprusso/toqito/pull/155)

- Thanks to @georgios-ts for fixing a bug in `sk_norm`.
  [PR-156](https://github.com/vprusso/toqito/pull/156)

- Thanks to @georgios-ts for fixing a bug in `is_unitary` for channels.
  [PR-164](https://github.com/vprusso/toqito/pull/164)

- Thanks to @georgios-ts for fixing an issue in `choi_to_kraus` for non-CP maps.
  [PR-162](https://github.com/vprusso/toqito/pull/162)

- Thanks for @georgios-ts for adding the `channel_dim` feature to allow one to
  extract the dimensions of a quantum channel.
  [PR-161](https://github.com/vprusso/toqito/pull/161)

- Thanks to @AmanieOxana for integrating the completely bounded trace and
  spectral norms. [PR-157](https://github.com/vprusso/toqito/pull/157)

- Thanks to @ErikQQY for refactoring and simplifying the partial trace and
  partial transpose functions.
  [PR-158](https://github.com/vprusso/toqito/pull/158)

- Thanks to @purva-thakre for adding SDPs for calculating the fidelity of
  separability in `fidelity_of_separability`
  [PR-165](https://github.com/vprusso/toqito/pull/165)

- Thanks to @elads888 for adding the `is_diagonally_dominant` feature to allow
  one to determine if a matrix is diagonally dominant.
  [PR-176](https://github.com/vprusso/toqito/pull/176)

- Thanks to @epelaaez for adding the `bures_angle` feature.
  [PR-181](https://github.com/vprusso/toqito/pull/181)

- Refactor of `state_exclusion` to take advantage of the `picos` SDP solver with
  to return both the optimal value and optimal measurements.
  [PR-182](https://github.com/vprusso/toqito/pull/182)

- Adding feature to determine if a given matrix is circulant with
  `is_circulant`. [PR-184](https://github.com/vprusso/toqito/pull/184)

  ## 1.0.7

- Numerous improvements to linting and overall improvement in project structure
  ([PR-190](https://github.com/vprusso/toqito/pull/190), [PR-192](https://github.com/vprusso/toqito/pull/192),
  [PR-193](https://github.com/vprusso/toqito/pull/193), [PR-195](https://github.com/vprusso/toqito/pull/195),
  [PR-200](https://github.com/vprusso/toqito/pull/200), [PR-205](https://github.com/vprusso/toqito/pull/205),
  [PR-206](https://github.com/vprusso/toqito/pull/206) [PR-212](https://github.com/vprusso/toqito/pull/212),
  [PR-223](https://github.com/vprusso/toqito/pull/223), and others). Thanks to @purva-thakre for a bunch of work here!

- Improved test coverage in channel fidelity([PR-232](https://github.com/vprusso/toqito/pull/232)), CB norm ([PR-234](https://github.com/vprusso/toqito/pull/234)), and KP-norm ([PR-246](https://github.com/vprusso/toqito/pull/246)), and more. Thanks to @purva-thakre!

- Fixed RTD failing builds. [PR-253](https://github.com/vprusso/toqito/pull/253).

- Optimizations to checking if set of vectors are mutually orthogonal. Improvement from O(n^2) to O(n). [PR-250](https://github.com/vprusso/toqito/pull/250).

- Feature to generate a set of mutually unbiased bases for a given prime dimension. [PR-272](https://github.com/vprusso/toqito/pull/272).

- Feature to check whether a given collection of quantum states described as vectors are antidistinguishable. [PR-271](https://github.com/vprusso/toqito/pull/271).

- Feature to check whether a set of quantum states constitute an unextendible product basis or not. (thanks to
  @Newtech66 and @smtsjhr).

- Feature to check if matrix is totally positive [PR-274](https://github.com/vprusso/toqito/pull/274)

- Fix: Bug in state exclusion code [PR-292](https://github.com/vprusso/toqito/pull/292)

- Feature: Determine whether a collection of vectors are linearly independent or not [PR-324](https://github.com/vprusso/toqito/pull/324)

- Feature: Adding the BB84 states [PR-327](https://github.com/vprusso/toqito/pull/327)

- Feature: Integrating the functionality to determine whether a collection of states comprise an unextendible product basis [PR-328](https://github.com/vprusso/toqito/pull/328)

- Feature: Adding the PBR states [PR-382](https://github.com/vprusso/toqito/pull/384)

- Feature: Adding cyclic permutation (thanks to @Bchass) [PR-390](https://github.com/vprusso/toqito/pull/390)

- Feature: Adding support for density matrix and vector input for state distinguishability and exclusion [PR-462](https://github.com/vprusso/toqito/pull/462)

## 1.0.8

- Patch for 1.0.7 as prior version was not including new files.