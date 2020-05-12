Getting Started Guide
=====================

Installing
------------------

The preferred way to install the :code:`toqito` package is via :code:`pip`

::

    pip install toqito

Alternatively, to install, you may also run the following command from the
top-level package directory.

::

    python setup.py install


Testing
-------

The :code:`nose` module is used for testing. To run the suite of tests for
:code:`toqito`, run the following command in the root directory of this project:

::

    nosetests --with-coverage --cover-erase --cover-package toqito

One may also use the :code:`pytest` module for testing as well:

::

    pytest tests/

Contributing
------------

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
`contributing guide <https://github.com/vprusso/toqito/blob/master/.github/CONTRIBUTING.md>`_.

Citing
------

You can cite :code:`toqito` using the following
DOI: `10.5281/zenodo.3699578 <https://zenodo.org/record/3699578>`_

If you are using the :code:`toqito` software package in research work, please
include an explicit mention of toqito in your publication. Something along the
lines of:

    To solve problem "X" we used `toqito`; a package for studying certain
    aspects of quantum information.

A BibTeX entry that you can use to cite :code:`toqito` is provided here:

::

    @misc{toqito,
       author       = {Vincent Russo},
       title        = {toqito: A {P}ython toolkit for quantum information, version 0.0.2},
       howpublished = {\url{https://github.com/vprusso/toqito}},
       month        = Mar,
       year         = 2020,
       doi          = {10.5281/zenodo.3699578}
     }
