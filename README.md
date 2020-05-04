# ToQITo (Theory of Quantum Information Toolkit)

The `toqito` package is an open source library for studying various objects in 
quantum information, namely, states, channels, and measurements.

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
[![Coverage Status](https://coveralls.io/repos/github/vprusso/toqito/badge.svg?branch=master?style=plastic)](https://coveralls.io/github/vprusso/toqito?branch=master)
[![DOI](https://zenodo.org/badge/235493396.svg?style=plastic)](https://zenodo.org/badge/latestdoi/235493396)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?styl=plastic)](https://github.com/psf/black)
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

The documentation for `toqito`, along with specific examples and tutorials of 
usage are provided on the 
[documentation page](https://toqito.readthedocs.io/en/latest/).

## Testing

The `nose` module is used for testing. To run the suite of tests for `toqito`,
run the following command in the root directory of this project.

    nosetests --with-coverage --cover-erase --cover-package toqito

## Citing

You can cite `toqito` using the following DOI: 
10.5281/zenodo.3699578

If you are using the `toqito` software package in research work, please include
an explicit mention of `toqito` in your publication. Something along the lines
of:

```
To solve problem "X" we used `toqito`; a package for studying certain
aspects of quantum information.
```

A BibTeX entry that you can use to cite `toqito` is provided here:

```
 @misc{toqito,
   author       = {Vincent Russo},
   title        = {toqito: A {P}ython toolkit for quantum information, version 0.0.2},
   howpublished = {\url{https://github.com/vprusso/toqito}},
   month        = Mar,
   year         = 2020,
   doi          = {10.5281/zenodo.3699578}
 }
```

## License

[MIT License](http://opensource.org/licenses/mit-license.php>)
