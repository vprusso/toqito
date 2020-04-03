# ToQITo (Theory of Quantum Information Toolkit)

The `toqito` package is a Python package for studying various aspects of
quantum information theory. Specifically, `toqito` focuses on providing
numerical tools to study problems pertaining to entanglement theory, nonlocal
games, matrix analysis, and other aspects of quantum information that are often
associated with computer science. 

[![build status](http://img.shields.io/travis/vprusso/toqito.svg)](https://travis-ci.org/vprusso/toqito)
[![Coverage Status](https://coveralls.io/repos/github/vprusso/toqito/badge.svg?branch=master)](https://coveralls.io/github/vprusso/toqito?branch=master)
[![DOI](https://zenodo.org/badge/235493396.svg)](https://zenodo.org/badge/latestdoi/235493396)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

`toqito` aims to fill the needs of quantum information researchers who want
numerical and computational tools for manipulating quantum states,
measurements, and channels. It can also be used as a tool to enhance the
experience of students and instructors in classes pertaining to quantum
information. 

The inspiration for this package is heavily influenced by the
[QETLAB](http://www.qetlab.com) package in MATLAB by Nathaniel Johnston.  Many
of the functions found here are direct ports of those functions converted into
Python code.

## Installation

The preferred way to install the `toqito` package is via `pip`:

```
pip install toqito
```

Alternatively, to install, you may also run the following command from the
top-level package directory.

```
python setup.py install
```

## Examples

All of the examples can be found in the form of 
[Python Jupyter notebook tutorials](https://github.com/vprusso/toqito/tree/master/docs/notebooks)

## Usage

The following lists the current functionality of `toqito`. Each bullet item 
currently links to or will link to a Jupyter notebook file that showcases the
usage.

Filling in the jupyter notebook example files lags behind the features
presently offered in `toqito` and will be periodically updated as time allows.

### Entanglement

Calculate various quantities of interest pertaining to entanglement.

  - [concurrence](https://github.com/vprusso/toqito/blob/master/docs/notebooks/entanglement/concurrence.ipynb): Computes the concurrence for a bipartite system.
  - [negativity](https://github.com/vprusso/toqito/blob/master/docs/notebooks/entanglement/negativity.ipynb): Computes the negativity of a bipartite density matrix.

### Matrix

#### Matrices

  - [clock](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/clock.ipynb): Generates the clock matrix.
  - [fourier](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/fourier.ipynb): Generate unitary matrix that implements the quantum Fourier transform.
  - [gell_mann](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/gell_mann.ipynb): Produces a Gell-Mann operator.
  - [gen_gell_man](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/gen_gell_mann.ipynb): Produces a generalized Gell-Mann operator.
  - [gen_pauli](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/gen_pauli.ipynb): Produces a generalized Pauli operator (sometimes called a Weyl operator).
  - [iden](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/iden.ipynb): Computes a sparse or full identity matrix.
  - [pauli](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/pauli.ipynb): Produces a Pauli operator.
  - [shift](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/matrices/shift.ipynb): Generates the shift matrix.

#### Operations

  - tensor: Kronecker tensor product of two or more matrices.
  - [vec](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/operations/vec.ipynb): Computes the vec representation of a given matrix.

#### Properties

  - [is_density](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_density.ipynb): Determines whether or not a matrix is a density matrix.
  - [is_diagonal](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_diagonal.ipynb): Determines whether or not a matrix is diagonal.
  - [is_hermitian](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_hermitian.ipynb): Determines whether or not a matrix is Hermitian.
  - [is_normal](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_normal.ipynb): Determines whether or not a matrix is normal.
  - [is_pd](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_pd.ipynb): Determines whether or not a matrix is positive definite.
  - [is_projection](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_projection.ipynb): Determines whether or not a matrix is a projection matrix.
  - [is_psd](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_psd.ipynb): Determines whether or not a matrix is positive semidefinite.
  - [is_square](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_square.ipynb): Determines whether or not a matrix is square.
  - [is_symmetric](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_symmetric.ipynb): Determines whether or not a matrix is symmetric.
  - [is_unitary](https://github.com/vprusso/toqito/blob/master/docs/notebooks/matrix/properties/is_unitary.ipynb): Determines whether or not a matrix is unitary.

### Measure

  - [is_povm](https://github.com/vprusso/toqito/blob/master/docs/notebooks/measure/is_povm.ipynb): Determines if a set of matrices are valid measurements operators.

### Nonlocal games

  - [two_player_quantum_lower_bound](https://github.com/vprusso/toqito/blob/master/docs/notebooks/nonlocal_games/nonlocal_games/two_player_quantum_lower_bound.ipynb): Computes a lower bound on the quantum value of a nonlocal game.

#### Bit commitment

#### Coin flipping

  - weak_coin_flipping: Weak coin flipping protocol.

#### Die rolling

#### Extended nonlocal games

#### Hedging

  - [hedging_value](https://github.com/vprusso/toqito/blob/master/docs/notebooks/nonlocal_games/hedging/hedging_value.ipynb): Semidefinite programs for obtaining values of quantum hedging scenarios.

#### XOR games

  - [xor_game_value](https://github.com/vprusso/toqito/blob/master/docs/notebooks/nonlocal_games/xor_games/xor_game_value.ipynb): Compute the classical or quantum value of a two-player nonlocal XOR game.

### Permutations

  - [antisymmetric_projection](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/antisymmetric_projection.ipynb): Produces the projection onto the antisymmetric subspace.
  - [perm_sign](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/perm_sign.ipynb): Computes the sign of a permutation.
  - [permutation_operator](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/permutation_operator.ipynb): Produces a unitary operator that permutes subsystems.
  - [permute_systems](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/permute_systems.ipynb): Permutes subsystems within a state or operator.
  - [swap](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/swap.ipynb): Swaps two subsystems within a state or operator.
  - [swap_operator](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/swap_operator.ipynb): Produces a unitary operator that swaps two subsystems.
  - [symmetric_projection](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/symmetric_projection.ipynb): Produces the projection onto the symmetric subspace.
  - [unique_perms](https://github.com/vprusso/toqito/blob/master/docs/notebooks/perms/unique_perms.ipynb): Compute all distinct permutations of a given vector.

### Random

  - [random_density_matrix](https://github.com/vprusso/toqito/blob/master/docs/notebooks/random/random_density_matrix.ipynb): Generates a random density matrix.
  - [random_povm](https://github.com/vprusso/toqito/blob/master/docs/notebooks/random/random_povm.ipynb): Generate a random set of positive-operator-valued measurements (POVMs).
  - [random_state_vector](https://github.com/vprusso/toqito/blob/master/docs/notebooks/random/random_state_vector.ipynb): Generates a random pure state vector.
  - [random_unitary](https://github.com/vprusso/toqito/blob/master/docs/notebooks/random/random_unitary.ipynb): Generates a random unitary or orthogonal matrix.

### State

#### Distance

  - bures_distance: Computes the Bures distance of two density matrices.
  - bures_metric: Computes the Bures metric between two density matrices.
  - [entropy](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/distance/entropy.ipynb): Computes the von Neumann or Rényi entropy of a density matrix.
  - [fidelity](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/distance/fidelity.ipynb): Computes the fidelity of two density matrices.
  - [helstrom_holevo](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/distance/helstorm_holevo.ipynb): Computes the Helstrom-Holevo distance between two density matrices.
  - [purity](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/distance/purity.ipynb): Computes the purity of a quantum state.
  - super_fidelity: Computes the super-fidelity of two density matrices.
  - [trace_distance](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/distance/trace_distance.ipynb): Computes the trace distance of two matrices.
  - [trace_norm](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/distance/trace_norm.ipynb): Computes the trace norm of a matrix.

#### Operations

  - [pure_to_mixed](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/operations/pure_to_mixed.ipynb): Converts a state vector or density matrix to a density matrix.
  - [schmidt_decomposition](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/operations/schmidt_decomposition.ipynb): Computes the Schmidt decomposition of a bipartite vector.
  - [schmidt_rank](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/operations/schmidt_rank.ipynb): Computes the Schmidt rank of a bipartite vector.

#### Optimizations
  - conclusive_state_exclusion: Calculates probability of conclusive single state exclusion.
  - ppt_distinguishability: Calculates probability of distinguishing via PPT measurements.
  - state_cloning: Calculate the optimal probability of cloning a quantum state.
  - state_discrimination: Calculates probability of state discrimination.
  - state_distance: Distinguish a set of quantum states.
  - unambiguous_state_exclusion: Calculates probability of unambiguous state exclusion.

#### Properties

  - [is_mixed](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/properties/is_mixed.ipynb): Determines if state is mixed.
  - is_mutually_unbiased_basis: Check if list of vectors constitute a mutually unbiased basis.
  - [is_ppt](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/properties/is_ppt.ipynb): Determines whether or not a matrix has positive partial transpose.
  - is_product_vector: Determines if a pure state is a product vector.
  - [is_pure](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/properties/is_pure.ipynb): Determines if a state is pure or a list of states are pure.
  - is_quantum_latin_square: Check if list of vectors constitute a quantum Latin square.

#### States

  - [bell](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/bell.ipynb): Produces a Bell state.
  - [chessboard](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/chessboard.ipynb): Produces a chessboard state.
  - [domino](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/domino.ipynb): Produces a Domino state.
  - [gen_bell](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/gen_bell.ipynb): Produces a generalized Bell state.
  - [ghz](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/ghz.ipynb): Generates a (generalized) GHZ state.
  - [gisin](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/gisin.ipynb): Generates a 2-qubit Gisin state.
  - [horodecki](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/horodecki.ipynb): Produces a Horodecki_state.
  - [isotropic](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/isotropic.ipynb): Produces an isotropic state.
  - [max_entangled](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/max_entangled.ipynb): Produces a maximally entangled bipartite pure state.
  - [max_mixed](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/max_mixed.ipynb): Produces the maximally mixed state.
  - [w_state](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/w_state.ipynb): Generates a (generalized) W-state.
  - [werner](https://github.com/vprusso/toqito/blob/master/docs/notebooks/state/states/werner.ipynb): Produces a Werner state.

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

## License

[MIT License](http://opensource.org/licenses/mit-license.php>)
