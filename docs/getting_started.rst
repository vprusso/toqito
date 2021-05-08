Getting started
===============

Installing
^^^^^^^^^^

1. Ensure you have Python 3.7 or greater installed on your machine.

2. Consider using a `virtual environment <https://packaging.python.org/guides/installing-using-pip-and-virtualenv/>`_.


3. The preferred way to install the :code:`toqito` package is via :code:`pip`.

.. code-block:: bash

    pip install toqito

Alternatively, to install, you may also run the following command from the
top-level package directory.

.. code-block:: bash

    python setup.py install

Installing BLAS/LAPACK
^^^^^^^^^^^^^^^^^^^^^^

The :code:`toqito` module makes heavy use of the :code:`cvxpy` module for solving various
convex optimization problems that naturally arise for certain problems in
quantum information. The installation instructions for :code:`cvxpy` may be found on
the project's `installation page <https://www.cvxpy.org/install/index.html>`_.

As a dependency for many of the solvers, you will need to ensure you have the
BLAS and LAPACK mathematical libraries installed on your machine. If you have
:code:`numpy` working on your machine, it is likely that you already have these
libraries on your machine. The :code:`cvxpy` module provides many different solvers
to select from for solving SDPs. We tend to use the
`SCS <https://github.com/cvxgrp/scs>`_ solver. Ensure that you have the :code:`scs`
Python module installed and built for your machine.

Testing
^^^^^^^

The :code:`pytest` module is used for testing. In order to run and :code:`pytest`, you will need to ensure it is
installed on your machine. Consult the `pytest <https://docs.pytest.org/en/latest/>`_ website for more information. To
run the suite of tests for :code:`toqito`, run the following command in the root directory of this project:

.. code-block:: bash

    pytest --cov-report term-missing --cov=toqito tests/

Contributing
^^^^^^^^^^^^

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
`contributing guide <https://github.com/vprusso/toqito/blob/master/.github/CONTRIBUTING.md>`_.

Reporting Issues
^^^^^^^^^^^^^^^^

Please report any issues you encounter via the
`issue template <https://github.com/vprusso/toqito/blob/master/.github/ISSUE_TEMPLATE.md>`_.

Citing
^^^^^^

You can cite :code:`toqito` using the following DOI: `10.5281/zenodo.4743211 <https://zenodo.org/record/4743211>`_.

If you are using the :code:`toqito` software package in research work, please
include an explicit mention of :code:`toqito` in your publication. Something
along the lines of:

    To solve problem "X" we used `toqito`; a package for studying certain
    aspects of quantum information.

A BibTeX entry that you can use to cite :code:`toqito` is provided here:

.. code-block:: bash

    @misc{toqito,
       author       = {Vincent Russo},
       title        = {toqito: A {P}ython toolkit for quantum information, version 1.0.0},
       howpublished = {\url{https://github.com/vprusso/toqito}},
       month        = Mar,
       year         = 2021,
       doi          = {10.5281/zenodo.4743211}
     }
