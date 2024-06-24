[![build status](https://github.com/vprusso/toqito/actions/workflows/build-test-actions.yml/badge.svg?style=plastic)](https://github.com/vprusso/toqito/actions/workflows/build-test-actions.yml)
[![doc status](https://readthedocs.org/projects/toqito/badge/?version=latest&style=plastic)](https://toqito.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/vprusso/toqito/branch/master/graph/badge.svg?style=plastic)](https://codecov.io/gh/vprusso/toqito)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4743211.svg)](https://doi.org/10.5281/zenodo.4743211)
[![Downloads](https://static.pepy.tech/personalized-badge/toqito?style=platic&period=total&units=none&left_color=black&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/toqito)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=plastic)](http://unitary.fund)


<p align="center">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://github.com/vprusso/toqito/raw/cfb62c4a5ce04b782f64229e7acd2b1c97f09801/docs/figures/logo.svg" width="60%">
   <img src="https://github.com/vprusso/toqito/raw/cfb62c4a5ce04b782f64229e7acd2b1c97f09801/docs/figures/logo.svg" width="60%">
 </picture>
 </p>


# toqito: Theory of Quantum Information Toolkit

The `toqito` package is an open-source Python library for studying various
objects in quantum information, namely, states, channels, and measurements.

<p align="center">
  <a href="https://toqito.readthedocs.io/en/latest/">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

Specifically, `toqito` focuses on providing numerical tools to study problems
about entanglement theory, nonlocal games, matrix analysis, and other
aspects of quantum information that are often associated with computer science.

`toqito` aims to fill the needs of quantum information researchers who want
numerical and computational tools for manipulating quantum states,
measurements, and channels. It can also be used as a tool to enhance the
experience of students and instructors in classes about quantum
information.


## Getting Started

toqito is available via [PyPi](https://pypi.org/project/toqito/) for Linux, and macOS, with support for Python 3.10 to 3.12.

```console
(venv) $ pip install toqito
```

The following code gives an example on the usage:

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

**Detailed documentation on all available methods, options, and input formats is available at [ReadTheDocs](https://toqito.readthedocs.io/en/latest/).**

## Using

Full documentation along with specific examples and tutorials are provided here:
[https://toqito.readthedocs.io/](https://toqito.readthedocs.io/). 

More information can also be found on the following
[toqito homepage](https://vprusso.github.io/toqito/).

Chat with us in our `toqito` channel on [Discord](http://discord.unitary.fund/). 

## Testing

The `pytest` module is used for testing. To run the suite of tests for `toqito`,
run the following command in the root directory of this project.

```
pytest --cov-report term-missing --cov=toqito
```

## Citing

You can cite `toqito` using the following DOI:
10.5281/zenodo.4743211


If you are using the `toqito` software package in research work, please include
an explicit mention of `toqito` in your publication. Something along the lines
of:

```
To solve problem "X" we used `toqito`; a package for studying certain
aspects of quantum information.
```

A BibTeX entry that you can use to cite `toqito` is provided here:

```bib
@misc{toqito,
   author       = {Vincent Russo},
   title        = {toqito: A {P}ython toolkit for quantum information, version 1.0.0},
   howpublished = {\url{https://github.com/vprusso/toqito}},
   month        = May,
   year         = 2021,
   doi          = {10.5281/zenodo.4743211}
 }
```

## References

The `toqito` project has been used or referenced in the following works:

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2406.13430&color=inactive&style=flat-square)](https://arxiv.org/abs/2406.13430) Bandyopadhyay, Somshubhro and Russo, Vincent
"Distinguishing a maximally entangled basis using LOCC and shared entanglement", (2024).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2307.2551&color=inactive&style=flat-square)](https://arxiv.org/abs/2307.02551) Tavakoli, Armin and Pozas-Kerstjens, Alejandro and Brown, Peter and Araújo, Mateus
"Semidefinite programming relaxations for quantum correlations", (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2311.17047&color=inactive&style=flat-square)](https://arxiv.org/abs/2311.17047) Johnston, Nathaniel and Russo, Vincent and Sikora, Jamie
"Tight bounds for antidistinguishability and circulant sets of pure quantum states", (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2308.15579&color=inactive&style=flat-square)](https://arxiv.org/abs/2308.15579) Pelofske, Elijah and Bartschi, Andreas and Eidenbenz, Stephan and Garcia, Bryan and Kiefer, Boris
"Probing Quantum Telecloning on Superconducting Quantum Processors", (2023).
 
- [![a](https://img.shields.io/static/v1?label=arXiv&message=2303.07911&color=inactive&style=flat-square)](https://arxiv.org/abs/2303.07911) Philip, Aby and Rethinasamy, Soorya and Russo, Vincent and Wilde, Mark. 
"Quantum Steering Algorithm for Estimating Fidelity of Separability.", Quantum 8, 1366, (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2302.09401&color=inactive&style=flat-square)](https://arxiv.org/abs/2302.09401) Miszczak, Jarosław Adam. 
"Symbolic quantum programming for supporting applications of quantum computing technologies.", (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2306.09444&color=inactive&style=flat-square)](https://arxiv.org/abs/2306.09444) Casalé, Balthazar and Di Molfetta, Giuseppe and Anthoine, Sandrine and Kadri, Hachem. 
"Large-Scale Quantum Separability Through a Reproducible Machine Learning Lens.", (2023).

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2206.08313&color=inactive&style=flat-square)](https://arxiv.org/abs/2206.08313) Russo, Vincent and Sikora, Jamie "Inner products of pure states and their antidistinguishability", Physical Review A, Vol. 107, No. 3, (2023).

## Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview of how to contribute can be found in the
[contributing guide](https://toqito.readthedocs.io/en/latest/getting_started.html#contributing).


## License

[MIT License](http://opensource.org/licenses/mit-license.php>)
