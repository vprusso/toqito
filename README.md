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

## Examples

All of the examples can be found in the form of 
[Python Jupyter notebook tutorials](https://github.com/vprusso/toqito/tree/master/docs/tutorials)

## Usage

The following lists the current functionality of `toqito`. Each bullet item 
currently links to or will link to a Jupyter notebook file that showcases the
usage.

Filling in the jupyter notebook example files lags behind the features
presently offered in `toqito` and will be periodically updated as time allows.

### Entanglement

Calculate various quantities of interest pertaining to entanglement.

  - Concurrence: Computes the concurrence for a bipartite system.
  - Negativity: Computes the negativity of a bipartite density matrix.

### Matrix

#### Matrices

  - [Clock matrix](https://github.com/vprusso/toqito/blob/master/docs/tutorials/matrix/matrices/clock_matrix.ipynb): Generates the clock matrix.
  - Fourier matrix: Generate unitary matrix that implements the quantum Fourier transform.
  - Gell-Mann matrix: Produces a Gell-Mann operator.
  - Generalized Gell-Mann matrices: Produces a generalized Gell-Mann operator.
  - Generalized Pauli matrices: Produces a generalized Pauli operator (sometimes called a Weyl operator).
  - Identity matrix: Computes a sparse or full identity matrix.
  - Pauli matrices: Produces a Pauli operator.
  - [Shift matrix](https://github.com/vprusso/toqito/blob/master/docs/tutorials/matrix/matrices/shift_matrix.ipynb): Generates the shift matrix.

#### Operations

  - Tensor product: Kronecker tensor product of two or more matrices.
  - Vec: Computes the vec representation of a given matrix.

#### Properties

  - Density matrix: Determines whether or not a matrix is a density matrix.
  - Diagonal matrix: Determines whether or not a matrix is diagonal.
  - Hermitian matrix: Determines whether or not a matrix is Hermitian.
  - Normal matrix: Determines whether or not a matrix is normal.
  - Positive-definite matrix: Determines whether or not a matrix is positive definite.
  - Projection matrix: Determines whether or not a matrix is a projection matrix.
  - Positive-semidefinite matrix: Determines whether or not a matrix is positive semidefinite.
  - Square matrix: Determines whether or not a matrix is square.
  - Symmetric matrix: Determines whether or not a matrix is symmetric.
  - Unitary matrix: Determines whether or not a matrix is unitary.

### Measure

  - Is measurement: Determines if a set of matrices are valid measurements operators.

### Nonlocal games

  - Two-player quantum value lower bound: Computes a lower bound on the quantum value of a nonlocal game.

#### Bit commitment

#### Coin flipping

  - Weak coin flipping: Weak coin flipping protocol.

#### Die rolling

#### Extended nonlocal games

#### Hedging

  - Hedging value: Semidefinite programs for obtaining values of quantum hedging scenarios.

#### XOR games

  - XOR game value: Compute the classical or quantum value of a two-player nonlocal XOR game.

### Permutations

  - Antisymmetric projection: Produces the projection onto the antisymmetric subspace.
  - Perfect matchings: Gives all perfect matchings of N objects.
  - Perm sign: Computes the sign of a permutation.
  - Permutation operator: Produces a unitary operator that permutes subsystems.
  - Permute systems: Permutes subsystems within a state or operator.
  - Swap: Swaps two subsystems within a state or operator.
  - Swap operator: Produces a unitary operator that swaps two subsystems.
  - Symmetric projection: Produces the projection onto the symmetric subspace.
  - Unique perms: Compute all distinct permutations of a given vector.

### Random

  - Random density matrix: Generates a random density matrix.
  - Random POVM: Generate a random set of positive-operator-valued measurements (POVMs).
  - Random state vector: Generates a random pure state vector.
  - Random unitary: Generates a random unitary or orthogonal matrix.

### State

#### Distance

  - Bures distance: Computes the Bures distance of two density matrices.
  - Bures metric: Computes the Bures metric between two density matrices.
  - Entropy: Computes the von Neumann or Rényi entropy of a density matrix.
  - Fidelity: Computes the fidelity of two density matrices.
  - Purity: Computes the purity of a quantum state.
  - Super fidelity: Computes the super-fidelity of two density matrices.
  - Trace distance: Computes the trace distance of two matrices.
  - Trace norm: Computes the trace norm of a matrix.

#### Operations

  - Pure-to-mixed: Converts a state vector or density matrix to a density matrix.
  - Schmidt decomposition: Computes the Schmidt decomposition of a bipartite vector.
  - Schmidt rank: Computes the Schmidt rank of a bipartite vector.

#### Optimizations
  - Conclusive state exclusion: Calculates probability of conclusive single state exclusion.
  - State cloning: Calculate the optimal probability of cloning a quantum state.
  - State discrimination: Calculates probability of state discrimination.
  - State distance: Distinguish a set of quantum states.
  - Unambiguous state exclusion: Calculates probability of unambiguous state exclusion.
  
#### Properties

  - Is mixed: Determines if state is mixed.
  - Is mutually unbiased basis: Check if list of vectors constitute a mutually unbiased basis.
  - Is PPT: Determines whether or not a matrix has positive partial transpose.
  - Is product vector: Determines if a pure state is a product vector.
  - [Is pure](https://github.com/vprusso/toqito/blob/master/docs/tutorials/state/properties/is_pure.ipynb): Determines if a state is pure or a list of states are pure.
  - Is quantum Latin square: Check if list of vectors constitute a quantum Latin square.


#### States

  - Bell states: Produces a Bell state.
  - Chessboard state: Produces a chessboard state.
  - Domino states: Produces a Domino state.
  - Generalized Bell states: Produces a generalized Bell state.
  - GHZ states: Generates a (generalized) GHZ state.
  - Gisin states: Generates a 2-qubit Gisin state.
  - Horodecki states: Produces a Horodecki_state.
  - Isotropic states: Produces an isotropic state.
  - Maximally entangled states: Produces a maximally entangled bipartite pure state.
  - Maximally mixed states: Produces the maximally mixed state.
  - W-state: Generates a (generalized) W-state.
  - Werner state: Produces a Werner state.

### Super operators

  - Apply map: Applies a superoperator to an operator.
  - Choi map: Produces the Choi map or one of its generalizations.
  - Dephasing channel: Produces a dephasing channel.
  - Depolarizing channel: Produces a depolarizng channel.
  - Partial trace: Computes the partial trace of a matrix.
  - Partial transpose: Computes the partial transpose of a matrix.
  - Realignment: Computes the realignment of a bipartite operator.
  - Reduction map: Produces the reduction map.
 
## Testing

The `nose` module is used for testing. To run the suite of tests for `toqito`,
run the following command in the root directory of this project.

    nosetests --with-coverage --cover-erase --cover-package .


## License

[MIT License](http://opensource.org/licenses/mit-license.php>)

