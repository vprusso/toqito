# ![logo](./docs/figures/logo.svg "logo")

(Theory of Quantum Information Toolkit)

The `toqito` package is an open source Python library for studying various
objects in quantum information, namely, states, channels, and measurements.

Specifically, `toqito` focuses on providing numerical tools to study problems
pertaining to entanglement theory, nonlocal games, matrix analysis, and other
aspects of quantum information that are often associated with computer science.

`toqito` aims to fill the needs of quantum information researchers who want
numerical and computational tools for manipulating quantum states,
measurements, and channels. It can also be used as a tool to enhance the
experience of students and instructors in classes pertaining to quantum
information.


[![build status](http://img.shields.io/travis/vprusso/toqito.svg?style=plastic)](https://travis-ci.org/vprusso/toqito)
[![doc status](https://readthedocs.org/projects/toqito/badge/?version=latest&style=plastic)](https://toqito.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/vprusso/toqito/branch/master/graph/badge.svg?style=plastic)](https://codecov.io/gh/vprusso/toqito)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4743211.svg)](https://doi.org/10.5281/zenodo.4743211)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=plastic)](http://unitary.fund)

## Installing

The preferred way to install the `toqito` package is via `pip`:

```
pip install toqito
```

Alternatively, to install, you may also run the following command from the
top-level package directory.

```
python setup.py install
```

## Using

Full documentation along with specific examples and tutorials are provided
here: [https://toqito.readthedocs.io/](https://toqito.readthedocs.io/).

More information can also be found on the following
[toqito homepage](https://vprusso.github.io/toqito/).

## Testing

The `pytest` module is used for testing. To run the suite of tests for `toqito`,
run the following command in the root directory of this project.

```
pytest --cov-report term-missing --cov=toqito tests/
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

The `toqito` project has been used in the following works:
 
- Philip, Aby, Soorya Rethinasamy, Vincent Russo, and Mark M. Wilde. "Quantum Steering Algorithm for Estimating Fidelity of Separability." arXiv preprint [arXiv:2303.07911](https://arxiv.org/abs/2303.07911) (2023).

- Miszczak, Jarosław Adam. "Symbolic quantum programming for supporting applications of quantum computing technologies." arXiv preprint [arXiv:2302.09401](https://arxiv.org/abs/2302.09401) (2023).

## Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](https://github.com/vprusso/toqito/blob/master/.github/CONTRIBUTING.md).

## License

[MIT License](http://opensource.org/licenses/mit-license.php>)
