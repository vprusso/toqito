---
title: 'toqito: A Python package for quantum information'
tags:
  - Python
  - quantum information
  - quantum computing
  - entanglement
  - matrix analysis
authors:
  - name: Vincent Russo
    affiliation: 1 
affiliations:
 - name: ISARA Corporation
   index: 1
date: 30 March 2020
bibliography: paper.bib

---

# Summary

toqito is a Python package for studying various aspects of quantum information
theory. Specifically, toqito focuses on providing numerical tools to study
problems pertaining to entanglement theory, nonlocal games, matrix analysis,
and other aspects of quantum information that are often associated with
computer science. While there are many outstanding feature-rich Python packages
to study quantum information, they are often focused on applications pertaining
to physics[@johansson2013qutip] [@killoran2019strawberry], [@steiger2018projectq].
Other excellent software offerings that are closer in scope to toqito, such as
QETLAB[@johnston2016qetlab], are written in non-opensource languages and
therefore require the users to have access to costly licenses.

toqito aims to fill the needs of quantum information researchers who want
numerical and computational tools for manipulating quantum states,
measurements, and channels. It can also be used as a tool to enhance the
experience of students and instructors in classes pertaining to quantum
information. 

toqito possesses functions for fundamental operations including the partial
trace, partial tranpose, and others. toqito also makes heavy use of the
CVXPY[@diamond2016cvxpy] convex optimization module to solve various
semidefinite programs that pertain to problems arising in the study of nonlocal
games, state discrimination, and other problems in quantum information.

The toqito module is supported for Python 3.7 and makes use of many of the more
modern features of the language including f-strings, type hinting, and others.
toqito is released under the BSD 3-Clause License and is available from GitHub
and PyPI.

# Acknowledgements
This research is supported by the Unitary Fund [@zeng2019unitary]

# References

