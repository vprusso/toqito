# ToQITo (Theory of Quantum Information Toolkit)

Suite of Python tools that come in handy when considering various concepts in
quantum information.

[![build status](http://img.shields.io/travis/vprusso/toqito.svg)](https://travis-ci.org/vprusso/toqito)
[![Coverage Status](https://coveralls.io/repos/github/vprusso/toqito/badge.svg?branch=master)](https://coveralls.io/github/vprusso/toqito?branch=master)
[![DOI](https://zenodo.org/badge/235493396.svg)](https://zenodo.org/badge/latestdoi/235493396)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

  - concurrence: Computes the concurrence for a bipartite system.
  - negativity: Computes the negativity of a bipartite density matrix.

### Matrix

#### Matrices

  - [clock](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/clock.ipynb): Generates the clock matrix.
  - [fourier](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/fourier.ipynb): Generate unitary matrix that implements the quantum Fourier transform.
  - [gell_mann](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/gell_mann.ipynb): Produces a Gell-Mann operator.
  - gen_gell_man: Produces a generalized Gell-Mann operator.
  - gen_pauli: Produces a generalized Pauli operator (sometimes called a Weyl operator).
  - iden: Computes a sparse or full identity matrix.
  - pauli: Produces a Pauli operator.
  - [shift](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/shift.ipynb): Generates the shift matrix.

#### Operations

  - tensor: Kronecker tensor product of two or more matrices.
  - vec: Computes the vec representation of a given matrix.

#### Properties

  - is_density: Determines whether or not a matrix is a density matrix.
  - is_diagonal: Determines whether or not a matrix is diagonal.
  - is_hermitian: Determines whether or not a matrix is Hermitian.
  - is_normal: Determines whether or not a matrix is normal.
  - is_pd: Determines whether or not a matrix is positive definite.
  - is_projection: Determines whether or not a matrix is a projection matrix.
  - is_psd: Determines whether or not a matrix is positive semidefinite.
  - is_square: Determines whether or not a matrix is square.
  - is_symmetric: Determines whether or not a matrix is symmetric.
  - is_unitary: Determines whether or not a matrix is unitary.

### Measure

  - is_povm: Determines if a set of matrices are valid measurements operators.

### Nonlocal games

  - two_player_quantum_lower_bound: Computes a lower bound on the quantum value of a nonlocal game.

#### Bit commitment

#### Coin flipping

  - weak_coin_flipping: Weak coin flipping protocol.

#### Die rolling

#### Extended nonlocal games

#### Hedging

  - hedging_value: Semidefinite programs for obtaining values of quantum hedging scenarios.

#### XOR games

  - xor_game_value: Compute the classical or quantum value of a two-player nonlocal XOR game.

### Permutations

  - antisymmetric_projection: Produces the projection onto the antisymmetric subspace.
  - perfect_matchings: Gives all perfect matchings of N objects.
  - perm_sign: Computes the sign of a permutation.
  - permutation_operator: Produces a unitary operator that permutes subsystems.
  - permute_systems: Permutes subsystems within a state or operator.
  - swap: Swaps two subsystems within a state or operator.
  - swap_operator: Produces a unitary operator that swaps two subsystems.
  - symmetric_projection: Produces the projection onto the symmetric subspace.
  - unique_perms: Compute all distinct permutations of a given vector.

### Random

  - random_density_matrix: Generates a random density matrix.
  - random_povm: Generate a random set of positive-operator-valued measurements (POVMs).
  - random_state_vector: Generates a random pure state vector.
  - random_unitary: Generates a random unitary or orthogonal matrix.

### State

#### Distance

  - bures_distance: Computes the Bures distance of two density matrices.
  - bures_metric: Computes the Bures metric between two density matrices.
  - entropy: Computes the von Neumann or Rényi entropy of a density matrix.
  - fidelity: Computes the fidelity of two density matrices.
  - helstrom_holevo: Computes the Helstrom-Holevo distance between two density matrices.
  - purity: Computes the purity of a quantum state.
  - super_fidelity: Computes the super-fidelity of two density matrices.
  - trace_distance: Computes the trace distance of two matrices.
  - trace_norm: Computes the trace norm of a matrix.

#### Operations

  - pure_to_mixed: Converts a state vector or density matrix to a density matrix.
  - schmidt_decomposition: Computes the Schmidt decomposition of a bipartite vector.
  - schmidt_rank: Computes the Schmidt rank of a bipartite vector.

#### Optimizations
  - conclusive_state_exclusion: Calculates probability of conclusive single state exclusion.
  - ppt_distinguishability: Calculates probability of distinguishing via PPT measurements.
  - state_cloning: Calculate the optimal probability of cloning a quantum state.
  - state_discrimination: Calculates probability of state discrimination.
  - state_distance: Distinguish a set of quantum states.
  - unambiguous_state_exclusion: Calculates probability of unambiguous state exclusion.
  
#### Properties

  - is_mixed: Determines if state is mixed.
  - is_mutually_unbiased_basis: Check if list of vectors constitute a mutually unbiased basis.
  - is_ppt: Determines whether or not a matrix has positive partial transpose.
  - is_product_vector: Determines if a pure state is a product vector.
  - [is_pure](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/properties/is_pure.ipynb): Determines if a state is pure or a list of states are pure.
  - is_quantum_latin_square: Check if list of vectors constitute a quantum Latin square.


#### States

  - bell: Produces a Bell state.
  - chessboard: Produces a chessboard state.
  - domino: Produces a Domino state.
  - gen_bell: Produces a generalized Bell state.
  - ghz: Generates a (generalized) GHZ state.
  - gisin: Generates a 2-qubit Gisin state.
  - horodecki: Produces a Horodecki_state.
  - isotropic: Produces an isotropic state.
  - max_entangled: Produces a maximally entangled bipartite pure state.
  - max_mixed: Produces the maximally mixed state.
  - w_state: Generates a (generalized) W-state.
  - werner: Produces a Werner state.

### Super operators

  - apply_map: Applies a superoperator to an operator.
  - choi_map: Produces the Choi map or one of its generalizations.
  - dephasing_channel: Produces a dephasing channel.
  - depolarizing_channel: Produces a depolarizng channel.
  - partial_trace: Computes the partial trace of a matrix.
  - partial_transpose: Computes the partial transpose of a matrix.
  - realignment: Computes the realignment of a bipartite operator.
  - reduction_map: Produces the reduction map.
 
## Testing

The `nose` module is used for testing. To run the suite of tests for `toqito`,
run the following command in the root directory of this project.

    nosetests --with-coverage --cover-erase --cover-package toqito

## License

[MIT License](http://opensource.org/licenses/mit-license.php>)

