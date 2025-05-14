<p align="center">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://github.com/vprusso/toqito/raw/cfb62c4a5ce04b782f64229e7acd2b1c97f09801/docs/figures/logo.svg" width="60%">
   <img src="https://github.com/vprusso/toqito/raw/cfb62c4a5ce04b782f64229e7acd2b1c97f09801/docs/figures/logo.svg" width="60%">
 </picture>
 </p>


# |toqito⟩: Theory of Quantum Information Toolkit

[![build status](https://github.com/vprusso/toqito/actions/workflows/build-test-actions.yml/badge.svg)](https://github.com/vprusso/toqito/actions/workflows/build-test-actions.yml)
[![doc status](https://readthedocs.org/projects/toqito/badge/?version=latest)](https://toqito.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/vprusso/toqito/branch/master/graph/badge.svg)](https://codecov.io/gh/vprusso/toqito)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4743211.svg)](https://doi.org/10.5281/zenodo.4743211)
[![Downloads](https://static.pepy.tech/personalized-badge/toqito?style=platic&period=total&units=none&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/toqito)
[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)

The |toqito⟩ package is an open-source Python library for studying various
objects in quantum information, namely, states, channels, and measurements.

<p align="center">
  <a href="https://toqito.readthedocs.io/en/latest/">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

|toqito⟩ focuses on providing numerical tools to study problems
about entanglement theory, nonlocal games, matrix analysis, and other
aspects of quantum information that are often associated with computer science.

|toqito⟩ aims to fill the needs of quantum information researchers who want
numerical and computational tools for manipulating quantum states,
measurements, and channels. It can also be used as a tool to enhance the
experience of students and instructors in classes about quantum
information.

## Installing

|toqito⟩ is available via [PyPi](https://pypi.org/project/toqito/) for Linux, and macOS, with support for Python 3.10 to
3.12.

```sh
pip install toqito
```

## Examples

Full documentation, along with specific examples and tutorials, is provided here:
[https://toqito.readthedocs.io/](https://toqito.readthedocs.io/). 

Chat with us in our |toqito⟩ channel on [Discord](http://discord.unitary.fund/). 

### Example: Nonlocal games

[Nonlocal games](https://toqito.readthedocs.io/en/latest/tutorials.nonlocal_games.html) are a mathematical framework
that abstractly models a physical system. The CHSH game is a subtype of nonlocal game referred to as an XOR game that
characterizes the seminal [CHSH inequality](https://en.wikipedia.org/wiki/CHSH_inequality). 

For XOR games, there exist optimization problems (that are provided via |toqito⟩) that one can compute to attain the
optimal values of such games when the players use either a classical or quantum strategy.  

```python
# Calculate the classical and quantum value of the CHSH game.
import numpy as np
from toqito.nonlocal_games.xor_game import XORGame

# The probability matrix.
prob_mat = np.array([[1/4, 1/4], [1/4, 1/4]])

# The predicate matrix.
pred_mat = np.array([[0, 0], [0, 1]])

# Define CHSH game from matrices.
chsh = XORGame(prob_mat, pred_mat)

chsh.classical_value()
# 0.75
chsh.quantum_value()
# 0.8535533
```
Indeed, using a quantum strategy for the CHSH game gives the known optimal result of $\frac{1}{4}\left(2 +
\sqrt{2}\right) \approx 0.8535...$

### Example: Quantum state distinguishability

Quantum state distinguishability is a fundamental task in quantum information theory. Consider the set of four [Bell
states](https://en.wikipedia.org/wiki/Bell_state):

$$
\begin{equation}
    \begin{aligned}
        |\psi_0\rangle = \frac{1}{\sqrt{2}} \left(|00\rangle + |11\rangle\right), \quad
        |\psi_1\rangle = \frac{1}{\sqrt{2}} \left(|00\rangle - |11\rangle\right), \\ 
        |\psi_2\rangle = \frac{1}{\sqrt{2}} \left(|01\rangle + |10\rangle\right), \quad
        |\psi_3\rangle = \frac{1}{\sqrt{2}} \left(|01\rangle - |10\rangle\right).
    \end{aligned}
\end{equation}
$$

The optimal probability of globally distinguishing the four Bell states (assuming an equal weighting of probability) is
1 (i.e., it can be performed perfectly). However, under a more restrictive set of measurements (such as PPT measurement
operators), the optimal probability of distinguishing the four Bell states using PPT operators is 1/2.

|toqito⟩ offers a wide suite of functionality for computing the distinguishability of quantum states:

```python
from toqito.states import bell
from toqito.state_opt import state_distinguishability, ppt_distinguishability

# Define the set of states as the four Bell states:
states = [bell(0), bell(1), bell(2), bell(3)]

# Distinguishing four Bell states (global measurements): 0.9999999999767388
pos_res, _ = state_distinguishability(states)
print(f"Distinguishing four Bell states (global measurements): {pos_res}")

# Distinguishing four Bell states (PPT measurements): 0.5000000000098367
ppt_res, _ = ppt_distinguishability(states, subsystems=[0], dimensions=[2, 2])
print(f"Distinguishing four Bell states (PPT measurements): {ppt_res}")
```

Consult the [tutorials](https://toqito.readthedocs.io/en/latest/tutorials.html#quantum-state-distinguishability) for
additional examples and information.

## Testing

The `pytest` module is used for testing. To run the suite of tests for |toqito⟩,
run the following command in the root directory of this project.

```
pytest --cov-report term-missing --cov=toqito
```

## Citing

You can cite |toqito⟩ using the following DOI:
`10.5281/zenodo.4743211`


If you are using the |toqito⟩ software package in research work, please include
an explicit mention of |toqito⟩ in your publication. Something along the lines
of:

```
To solve problem "X", we used |toqito⟩; a package for studying certain
aspects of quantum information.
```

A BibTeX entry that you can use to cite |toqito⟩ is provided here:

```bib
@misc{toqito,
   author       = {Vincent Russo},
   title        = {toqito: A {P}ython toolkit for quantum information},
   howpublished = {\url{https://github.com/vprusso/toqito}},
   month        = May,
   year         = 2021,
   doi          = {10.5281/zenodo.4743211}
 }
```

## Research

The |toqito⟩ project is, first and foremost, a quantum information theory research tool. Consult the following [open
problems wiki
page](https://github.com/vprusso/toqito/wiki/Research-open-problems-in-quantum-information-theory-using-%7Ctoqito%E2%9F%A9)
for a list of certain solved and unsolved problems in quantum information theory in which |toqito⟩ could be potentially
helpful in probing. Feel free to add to this list and/or contribute solutions!

The |toqito⟩ project has been used or referenced in the following works:

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2311.17047&color=inactive&style=flat-square)](https://arxiv.org/abs/2311.17047) Johnston, Nathaniel and Russo, Vincent and Sikora, Jamie
*"Tight bounds for antidistinguishability and circulant sets of pure quantum states"*, Quantum 9, 1622, (2025).

- [![a](https://img.shields.io/static/v1?label=thesis&message=31639397&color=inactive&style=flat-square)](https://www.proquest.com/openview/eb0021dd3eb463b5fb12b7fc71d920eb/1?cbl=18750&diss=y&pq-origsite=gscholar) Philip, Aby
*"On Multipartite Entanglement and Its Use"*, (2024).

- [![a](https://img.shields.io/static/v1?message=report&color=inactive&style=flat-square)](https://ali-almasi.github.io/assets/projects_materials/Internship_report.pdf) Almasi, Ali
*"Quantum Guessing Games"*, (2024).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2409.08705&color=inactive&style=flat-square)](https://arxiv.org/abs/2409.08705)
Gupta, Tathagata and Mushid, Shayeef and Russo, Vincent and Bandyopadhyay, Somshubhro
*"Optimal discrimination of quantum sequences"*, Physical Review A, 110, 062426, (2024).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2406.13430&color=inactive&style=flat-square)](https://arxiv.org/abs/2406.13430) Bandyopadhyay, Somshubhro and Russo, Vincent
*"Distinguishing a maximally entangled basis using LOCC and shared entanglement"*, Physical Review A 110, 042406, (2024).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2307.2551&color=inactive&style=flat-square)](https://arxiv.org/abs/2307.02551) Tavakoli, Armin and Pozas-Kerstjens, Alejandro and Brown, Peter and Araújo, Mateus
*"Semidefinite programming relaxations for quantum correlations"*, Reviews of Modern Physics, Volume 96, (2024).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2308.15579&color=inactive&style=flat-square)](https://arxiv.org/abs/2308.15579) Pelofske, Elijah and Bartschi, Andreas and Eidenbenz, Stephan and Garcia, Bryan and Kiefer, Boris
*"Probing Quantum Telecloning on Superconducting Quantum Processors"*, IEEE Transactions on Quantum Engineering, (2024).
 
- [![a](https://img.shields.io/static/v1?label=arXiv&message=2303.07911&color=inactive&style=flat-square)](https://arxiv.org/abs/2303.07911) Philip, Aby and Rethinasamy, Soorya and Russo, Vincent and Wilde, Mark. 
*"Quantum Steering Algorithm for Estimating Fidelity of Separability"*, Quantum 8, 1366, (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2302.09401&color=inactive&style=flat-square)](https://arxiv.org/abs/2302.09401) Miszczak, Jarosław Adam. 
*"Symbolic quantum programming for supporting applications of quantum computing technologies"*, Companion Proceedings of the 7th International Conference on the Art, Science, and Engineering of Programming, (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2306.09444&color=inactive&style=flat-square)](https://arxiv.org/abs/2306.09444) Casalé, Balthazar and Di Molfetta, Giuseppe and Anthoine, Sandrine and Kadri, Hachem. 
*"Large-Scale Quantum Separability Through a Reproducible Machine Learning Lens"*, (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2206.08313&color=inactive&style=flat-square)](https://arxiv.org/abs/2206.08313) Russo, Vincent and Sikora, Jamie *"Inner products of pure states and their antidistinguishability"*, Physical Review A, Vol. 107, No. 3, (2023).

## Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview of how to contribute can be found in the
[contributing guide](https://toqito.readthedocs.io/en/latest/contributing.html#contrib-guide-reference-label).

## License

[MIT License](http://opensource.org/licenses/mit-license.php>)
