---
title: 'toqito -- Theory of quantum information toolkit: A Python package for studying quantum information'
tags:
  - Python
  - quantum information
  - quantum computing
  - entanglement
  - nonlocal games
  - matrix analysis
authors:
  - name: Vincent Russo
    affiliation: 1 
affiliations:
 - name: ISARA Corporation
   index: 1
date: 05 May 2020
bibliography: paper.bib

---

# Summary

toqito is an open source library for studying various objects in quantum
information, namely, states, channels, and measurements. Specifically, toqito
focuses on providing numerical tools to study problems pertaining to
entanglement theory, nonlocal games, matrix analysis, and other aspects of
quantum information that are often associated with computer science. While
there are many outstanding feature-rich Python packages to study quantum
information, they are often focused on applications pertaining to
physics[@johansson2013qutip] [@killoran2019strawberry], [@steiger2018projectq].
Other excellent software offerings that are closer in scope to toqito, such as
QETLAB[@johnston2016qetlab], are written in non-opensource languages and
therefore require the users to have access to costly licenses.

toqito possesses functions for fundamental operations including the partial
trace, partial tranpose, and others. toqito also makes use of the
CVXPY[@diamond2016cvxpy] convex optimization module to solve various
semidefinite programs that pertain to problems arising in the study of nonlocal
games, state discrimination, and other problems in quantum information.
Specifically, toqito provides the ability to either directly calculate or
estimate the classical and quantum values of nonlocal games. toqito also
provides numerous functions for performing operations on and for determining
properties of quantum states, channels, and measurements. toqito provides
numerous functions for exploring measures of entanglement and properties of
entangled states. Support for generating random quantum states and measurement
operators is also provided. 

The toqito module is supported for Python 3.7 and makes use of many of the more
modern features of the language including f-strings, type hinting, and others.
toqito is available on GitHub (https://github.com/vprusso/toqito) and can be
installed via pip. Further information of features and uses can be found on the
documentation page (https://toqito.readthedocs.io/en/latest/).

# Acknowledgements
This research is supported by the Unitary Fund [@zeng2019unitary]

# References

