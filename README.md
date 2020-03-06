# ToQITo (Theory of Quantum Information Toolkit)

Suite of Python tools that come in handy when considering various concepts in
quantum information.

[![build status](http://img.shields.io/travis/vprusso/toqito.svg)](https://travis-ci.org/vprusso/toqito)
[![Coverage Status](https://coveralls.io/repos/github/vprusso/toqito/badge.svg?branch=master)](https://coveralls.io/github/vprusso/toqito?branch=master)
[![DOI](https://zenodo.org/badge/235493396.svg)](https://zenodo.org/badge/latestdoi/235493396)

The inspiration for this package is heavily influenced by the
[QETLAB](http://www.qetlab.com) package in MATLAB by Nathaniel Johnston.  Many
of the functions found here are direct ports of those functions converted into
Python code.

## Citing toqito

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
   title        = {toqito: A {P}ython toolkit for quantum information, version 0.1},
   howpublished = {\url{https://github.com/vprusso/toqito}},
   month        = Mar,
   year         = 2020,
   doi          = {10.5281/zenodo.3699578}
 }
```

## Usage

The following lists the current functionality of `toqito`. Each bullet item 
currently links to or will link to a Jupyter notebook file that showcases the
usage.

Filling in the jupyter notebook example files lags behind the features
presently offered in `toqito` and will be periodically updated as time allows.

### Entanglement

Calculate various quantities of interest pertaining to entanglement.

- Concurrence
- Negativity

### Matrix

#### Matrices

- [Clock matrix](https://github.com/vprusso/toqito/blob/master/jupyter_notebooks/matrix/matrices/clock_matrix.ipynb)
- Fourier matrix
- Gell-Mann matrix
- Generalized Gell-Mann matrices
- Generalized Pauli matrices
- Identity matrix
- Pauli matrices
- [Shift matrix](https://github.com/vprusso/toqito/blob/master/jupyter_notebooks/matrix/matrices/shift_matrix.ipynb)

#### Operations

- Tensor product
- Vec

#### Properties

- Density matrix
- Diagonal matrix
- Hermitian matrix
- Normal matrix
- Positive-definite matrix
- Positive-semidefinite matrix
- Square matrix
- Symmetric matrix
- Unitary matrix

### Measure

- Is measurement

### Nonlocal games

- Two-player quantum value lower bound

#### Bit commitment

#### Coin flipping

- Weak coin flipping

#### Die rolling

#### Extended nonlocal games

#### Hedging

- Hedging value

#### XOR games

- XOR game value

### Permutations

- Antisymmetric projection
- Perfect matchings
- Perm sign
- Permutation operator
- Permute systems
- Swap
- Swap Operator
- Symmetric projection
- Unique perms

### Random

- Random density matrix
- Random POVM
- Random state vector
- Random unitary

### State

#### Distance

- Bures distance
- Bures metric
- Entropy
- Fidelity
- Purity
- Super fidelity
- Trace distance
- Trace norm

#### Operations

- Pure-to-mixed
- Schmidt decomposition
- Schmidt rank

#### Optimizations

- State cloning
- State discrimination
- State distance
- State exclusion

#### Properties

- Is mixed
- Is PPT
- Is product vector
- Is pure

#### States

- Bell states
- Chessboard state
- Domino states
- Generalized Bell states
- GHZ states
- Gisin states
- Horodecki states
- Isotropic states
- Maximally entangled states
- W-state
- Werner state

### Super operators

- Apply map
- Choi map
- Dephasing channel
- Depolarizing channel
- Partial trace
- Partial transpose
- Realignment
- Reduction map
 
## Testing

The `nose` module is used for testing. To run the suite of tests for `toqito`,
run the following command in the root directory of this project.

    nosetests --with-coverage --cover-erase --cover-package .


## License

[MIT License](http://opensource.org/licenses/mit-license.php>)

