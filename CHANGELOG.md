# 0.5

- Fix: Bug in `swap`. Added test to cover bug found.

- Fix: Bug in `ppt_distinguishability` that prevented checking for higher
  dimensions. Added test to cover bug found.

- Fix: Bug in `ppt_distinguishability` that prevented checking 2-dimensional
  cases.

- Fix: Bug in `state_distinguishability`. Value returned was being multiplied
  by an unnecessary factor. Added other tests that would have caught this.

- Fix: Bug in `partial_trace`. Added test to cover bug found.

- Fix: Bug in `partial_transpose` for non-square matrices. Adding test cases to
  cover this previously failing case.
 
- Fix: Bug in `purity`. Squaring matrix was using `**2` instead of 
  `np.linalg.matrix_power` 

- Feature: Added in `symmetric_extension_hierarchy`. The function is a
  hierarchy of semidefinite programs that eventually converge to the separable
  value for distinguishing an ensemble of quantum states. The first level is
  just the standard PPT distinguishability probability. Computing higher levels
  of this hierarchy can become intractable quite quickly.

- Feature: Added in ability to perform both minimum-error and unambiguous state
  discrimination in `state_distinguishability.py`. Adding additional tests to
  cover this extra feature.
  
- Feature: Added `majorizes`; a function to determine whether a vector or matrix
  majorizes another. This is used as a criterion to determine whether a quantum
  state can be converted to another via LOCC.

- Enhancement: Adding further tests for `symmetric_projection`.

- Enhancement: Adding further tests for `ppt_distinguishability`.
