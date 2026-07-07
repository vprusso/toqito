---
name: Performance improvement
about: Suggest a way to make a function faster or more memory-efficient
title: ''
labels: enhancement, refactor
assignees: ''

---

## Which function?
*Path and symbol, e.g. `toqito/matrix_ops/partial_trace.py::partial_trace`.*

## What is inefficient?
*Describe the wasteful work: a redundant decomposition, a dense operation where structure is sparse, a Python loop that could be vectorized, an SDP rebuilt inside a loop, an `O(n^3)` where `O(n^2)` suffices, etc.*

## Proposed change
*The cheaper approach. If you have measured it, include a before/after timing and the input sizes used.*

## Correctness
*Confirm the change is behavior-preserving (same output up to floating-point roundoff), or note any edge case that differs. Existing tests should still pass.*
