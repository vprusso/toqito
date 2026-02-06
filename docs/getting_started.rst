.. _getting_started_reference-label:

===============
Getting started
===============

.. warning::
    Efficacy of :code:`|toqito⟩` has not been verified on Windows. 

----------
Installing
----------

1. Ensure you have Python 3.10 or greater installed on your machine or in
a virtual environment (`pyenv <https://github.com/pyenv/pyenv>`_, `pyenv tutorial <https://realpython.com/intro-to-pyenv/>`_).

.. note::
    On macOS, the :code:`cvxopt` dependency (pulled in by :code:`picos`) may need to be built from source
    if a pre-built wheel is not available for your Python version and macOS version. If you encounter a
    build error mentioning :code:`umfpack.h`, install the required system library with:

    .. code-block:: bash

        brew install suite-sparse

2. Consider using a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.
You can also use :code:`pyenv` with :code:`virtualenv` `to manage different Python versions <https://github.com/pyenv/pyenv-virtualenv>`_. 

3. The preferred way to install the :code:`|toqito⟩` package is via :code:`uv`, which keeps dependencies in sync with the
project's lockfile. An editable version of :code:`|toqito⟩` can be installed through the instructions provided
in the :ref:`contrib_guide_reference-label`.

If you prefer to not install an editable version of :code:`|toqito⟩`, use:

.. code-block:: bash

    (local_venv) pip install toqito

Above command will also install other additional dependencies for :code:`|toqito⟩`.  

The :code:`|toqito⟩` module makes heavy use of the :code:`cvxpy` module for solving various convex optimization problems
that naturally arise for certain problems in quantum information. The installation instructions for :code:`cvxpy` may be found on
the project's `installation page <https://www.cvxpy.org/install/index.html>`_. However these installation instructions
can be ignored as :code:`pip install toqito` will also install :code:`cvxpy` as a dependency.

.. note::
    macOS already ships with BLAS and LAPACK installed by default under the `Accelerate framework <https://developer.apple.com/documentation/accelerate/blas/>`_.

As a dependency for many of the solvers, you will need to ensure you have the :code:`BLAS` and :code:`LAPACK`
mathematical libraries installed on your machine. If you have :code:`numpy` working on your machine
(installed as a :code:`|toqito⟩` dependency), you already have these libraries on your machine. See NumPy `docs <https://numpy.org/doc/stable/building/blas_lapack.html>`_. If you don't,
:code:`BLAS` and :code:`LAPACK` can be installed using the following command:

.. code-block:: bash

    (For Linux) sudo apt-get install -y libblas-dev liblapack-dev

The :code:`cvxpy` module provides many different solvers to select from for solving SDPs. We tend to use the
`SCS <https://github.com/cvxgrp/scs>`_ solver. Ensure that you have the :code:`scs` Python module installed and built
for your machine. Again, this discussion can be ignored as :code:`pip install toqito` will also install :code:`SCS` as a
dependency.

------------
Contributing
------------

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the  :ref:`contrib_guide_reference-label`.

----------------
Reporting Issues
----------------

Please report any issues you encounter on `GitHub <https://github.com/vprusso/toqito/issues>`_.

------
Citing
------

You can cite :code:`|toqito⟩` using the following DOI: `10.5281/zenodo.4743211 <https://zenodo.org/record/4743211>`_.

If you are using the :code:`|toqito⟩` software package in research work, please
include an explicit mention of :code:`|toqito⟩` in your publication. Something
along the lines of:

.. code-block:: text

    To solve problem "X" we used `toqito`; a package for studying certain aspects of quantum information.

A BibTeX entry that you can use to cite :code:`|toqito⟩` is provided here:

.. code-block:: text

    @misc{toqito,
       author       = {Vincent Russo},
       title        = {toqito: A {P}ython toolkit for quantum information, version 1.0.0},
       howpublished = {\url{https://github.com/vprusso/toqito}},
       month        = Mar,
       year         = 2021,
       doi          = {10.5281/zenodo.4743211}
     }