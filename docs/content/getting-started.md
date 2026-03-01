# Getting started

!!! warning
    Efficiency of `|toqito⟩` has not been verified on Windows.

## Installing

1\. Ensure you have Python 3.11 or greater (up to 3.14) installed on your machine or
in a virtual environment ([pyenv](https://github.com/pyenv/pyenv),
[pyenv tutorial](https://realpython.com/intro-to-pyenv/)).

!!! note
    On macOS, the `cvxopt` dependency (pulled in by `picos`) may need to be
    built from source if a pre-built wheel is not available for your Python
    version and macOS version. If you encounter a build error mentioning 
    `umfpack.h`, install the required system library with:

    ``` bash
    brew install suite-sparse
    ```

2\. Consider using a [virtual
environment](https://docs.python.org/3/tutorial/venv.html). You can also
use `pyenv` with `virtualenv` [to manage different Python
versions](https://github.com/pyenv/pyenv-virtualenv).

3\. The preferred way to install the `|toqito⟩` package is via `uv`,
which keeps dependencies in sync with the project's lockfile. An
editable version of `|toqito⟩` can be installed through the instructions
provided in the [Contributing Guide](./contributing-guide.md).

If you prefer to not install an editable version of `|toqito⟩`, use:

``` bash
(local_venv) pip install toqito
```

Above command will also install other additional dependencies for
`|toqito⟩`.

The `|toqito⟩` module makes heavy use of the `cvxpy` module for solving
various convex optimization problems that naturally arise for certain
problems in quantum information. The installation instructions for
`cvxpy` may be found on the project's [installation
page](https://www.cvxpy.org/install/index.html). However these
installation instructions can be ignored as `pip install toqito` will
also install `cvxpy` as a dependency.

!!! note
    macOS already ships with BLAS and LAPACK installed by default under the
    [Accelerate
    framework](https://developer.apple.com/documentation/accelerate/blas/).


As a dependency for many of the solvers, you will need to ensure you
have the `BLAS` and `LAPACK` mathematical libraries installed on your
machine. If you have `numpy` working on your machine (installed as a
`|toqito⟩` dependency), you already have these libraries on your
machine. See NumPy
[docs](https://numpy.org/doc/stable/building/blas_lapack.html). If you
don't, `BLAS` and `LAPACK` can be installed using the following
command:

``` bash
(For Linux) sudo apt-get install -y libblas-dev liblapack-dev
```

The `cvxpy` module provides many different solvers to select from for
solving SDPs. We tend to use the [SCS](https://github.com/cvxgrp/scs)
solver. Ensure that you have the `scs` Python module installed and built
for your machine. Again, this discussion can be ignored as
`pip install toqito` will also install `SCS` as a dependency.

## Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
[Contributing Guide](./contributing-guide.md).

## Reporting Issues

Please report any issues you encounter on
[GitHub](https://github.com/vprusso/toqito/issues).

## Citing

You can cite `|toqito⟩` using the following DOI:
[10.5281/zenodo.4743211](https://zenodo.org/record/4743211).

If you are using the `|toqito⟩` software package in research work,
please include an explicit mention of `|toqito⟩` in your publication.
Something along the lines of:

``` text
To solve problem "X" we used `toqito`; a package for studying certain aspects of quantum information.
```

A BibTeX entry that you can use to cite `|toqito⟩` is provided here:

``` text
@​misc{toqito,
   author       = {Vincent Russo},
   title        = {toqito: A {P}ython toolkit for quantum information, version 1.0.0},
   howpublished = {\url{https://github.com/vprusso/toqito}},
   month        = Mar,
   year         = 2021,
   doi          = {10.5281/zenodo.4743211}
 }
```
