Getting Started Guide
=====================

Installing
------------------

The preferred way to install the `toqito` package is via `pip`

::

    pip install toqito

Alternatively, to install, you may also run the following command from the top-level package directory.

::

    python setup.py install


Testing
-------

The `nose` module is used for testing. To run the suite of tests for `toqito`,
run the following command in the root directory of this project:

::

    nosetests --with-coverage --cover-erase --cover-package toqito


Contributing
------------

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](https://github.com/vprusso/toqito/blob/master/.github/CONTRIBUTING.md)

Citing
------

You can cite `toqito` using the following DOI: 10.5281/zenodo.3699578

If you are using the `toqito` software package in research work, please include
an explicit mention of toqito in your publication. Something along the lines of:

    To solve problem "X" we used `toqito`; a package for studying certain
    aspects of quantum information.

A BibTeX entry that you can use to cite toqito is provided here:

::

    @misc{toqito,
       author       = {Vincent Russo},
       title        = {toqito: A {P}ython toolkit for quantum information, version 0.0.2},
       howpublished = {\url{https://github.com/vprusso/toqito}},
       month        = Mar,
       year         = 2020,
       doi          = {10.5281/zenodo.3699578}
     }
