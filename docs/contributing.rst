.. _contrib_guide_reference-label:

""""""""""""""""""
Contributing Guide
""""""""""""""""""

We welcome contributions from external contributors, and this document describes how to merge code changes into
:code:`toqito`. 


1. Make sure you have a `GitHub account <https://github.com/signup/free>`_.
2. `Fork <https://help.github.com/articles/fork-a-repo/>`_ this repository on GitHub.
3. On your local machine, `clone <https://help.github.com/articles/cloning-a-repository/>`_ your fork of the repository. You will
   have to install an editable version on your local machine. Instructions are provided below.


.. warning::
     It would be better to avoid an editable installation via :code:`pip` as :code:`poetry` is a better dependency resolver. 

4. As stated in :ref:`getting_started_reference-label`, ensure you have Python 3.10 or greater installed on your machine or in 
   a virtual environment (`pyenv <https://github.com/pyenv/pyenv>`_, `pyenv tutorial <https://realpython.com/intro-to-pyenv/>`_).
   Consider using a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.
   You can also use :code:`pyenv` with :code:`virtualenv` `to manage different Python
   versions <https://github.com/pyenv/pyenv-virtualenv>`_ or :code:`conda` to create virtual environments with `different Python
   versions <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments>`_.

5. Install :code:`poetry` using the following command. :code:`poetry` is a better dependency resolver than :code:`pip`.

.. code-block:: bash

    (local_venv) pip install poetry --upgrade pip

6. Now, navigate to your local clone of the :code:`toqito` repository as shown below.

.. code-block:: bash

    (local_venv) cd toqito

7. Use :code:`poetry` as shown below in the :code:`toqito` folder. This should install an editable version of :code:`toqito`
   alongside other development dependencies.

.. code-block:: bash

    (local_venv)~/toqito$ poetry install

You are now free to make the desired changes in your fork of :code:`toqito`. 

--------------
Making Changes
--------------

1.   Add some really awesome code to your local fork.  It's usually a 
     `good idea <http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/>`_
     to make changes on a 
     `branch <https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/>`_
     with the branch name relating to the feature you are going to add.

2.   When you are ready for others to examine and comment on your new feature,
     navigate to your fork of :code:`toqito` on GitHub and open a 
     `pull request(PR) <https://help.github.com/articles/using-pull-requests/>`_ . Note that
     after you launch a PR from one of your fork's branches, all subsequent commits to that branch will be added to the
     open pull request automatically.  Each commit added to the PR will be validated for mergability, compilation and
     test suite compliance; the results of these tests will be visible on the PR page.

3.   If you're adding a new feature, you must add test cases and documentation. See `Adding a new feature`_
     for a detailed checklist. 

4.   When the code is ready to go, make sure you run the test suite using :code:`pytest`, :code:`ruff`, etc.

5.   When you're ready to be considered for merging, comment on your PR that it is ready for a review
     to let the :code:`toqito` devs know that the changes are complete. The code will not be reviewed
     until you have commented so, the continuous integration workflow passes, and the primary developer approves the
     reviews.

-------
Testing
-------

A convenient way to verify if the installation procedure worked correctly, use `pytest` in the :code:`toqito` folder as
shown below.

.. code-block:: bash

    (local_venv)~/toqito$ pytest toqito/

The :code:`pytest` module is used for testing and :code:`pytest-cov` can be used to generate
coverage reports locally. In order to run and :code:`pytest`, you will need to ensure it is installed on your machine
along with :code:`pytest-cov`. If the editable installation process worked without any issues, both :code:`pytest` and
:code:`pytest-cov` should be installed in your local environment. 

If not, consult the `pytest <https://docs.pytest.org/en/latest/>`_  and
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ websites for additional options on code coverage reports.
For example, if your addition is not properly covered by tests, code coverage can be checked by using
:code:`--cov-report term-missing` options in :code:`pytest-cov`.

If you are making changes to :code:`toqito.some_module`, the corresponding tests should be in
:code:`toqito/some_module/tests`.


----------
Code Style
----------


We use :code:`ruff` to check for formatting issues. Consult the documentation for
`ruff <https://docs.astral.sh/ruff/tutorial/#getting-started>`_ for additional information.

Do not use an autoformatter like :code:`black` as the configuration settings for :code:`ruff` as specified in
`pyproject.toml <https://github.com/vprusso/toqito/blob/8606650b98608330c8b89414f7fb641992517ee4/pyproject.toml>`_
might be incompatible with the changes made by :code:`black`. This is discussed in detail at
`this link <https://docs.astral.sh/ruff/formatter/black/>`_.

Setting Up Pre-Commit Hooks
Pre-commit hooks ensure that code meets our formatting and linting standards before it is committed to the repository. Install the hooks with the following command.

.. code-block:: bash
   
   poetry run pre-commit install

This integrates ruff checks into your workflow, ensuring consistent code quality across the project. 

Additionaly, the commit-msg hook ensures adherence to the `Conventional Commits <https://www.conventionalcommits.org/>`_ format for all commit messages and helps maintain a standardized commit history.

.. code-block:: bash

    poetry run pre-commit install --hook-type commit-msg

------------------------
References in Docstrings
------------------------


If you are adding a new function, make sure the docstring of your function follows the formatting specifications
in `Code Style`_. A standard format for :code:`toqito` docstring is provided below:

.. code-block:: python
    
    def my_new_function(some_parameter: parameter_type) -> return_type:
        r"""One liner description of the new function.

            Detailed description of the function.

            Examples
            ==========
            Demonstrate how the function works with expected output.

            References
            ==========
            .. bibliography::
                :filter: docname in docnames
        
            :param name_of_parameter: Description of the parameter.
            :raises SomeError: Description for when the function raises an error.
            :return: Description of what the function returns.
                
        """

Use :code:`.. math::` mode for equations and use use :code:`:cite:some_ref` for some reference in the docstring. 

To add an attribution to a paper or a book, add your reference with :code:`some_ref` as the citation key to 
`refs.bib`.

Following is used in a docstring for the references to show up in the documentation build.

.. code-block:: text

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


--------------
Documentation
--------------


We use :code:`sphinx` to build the documentation and :code:`doctest` to test the examples in the documentation and function docstrings. 
To build the documentation locally, make sure :code:`sphinx` and :code:`furo` are installed when poetry was used to
install :code:`toqito`.

.. code-block:: bash

    (local_venv)~/toqito/docs$ make clean html

If you would prefer to decrease the amount of time taken by :code:`sphinx` to build the documentation locally, use :code:`make html`
instead.

A standard document has to follow the :code:`.rst` format.  For more information on :code:`sphinx` and
the documentation theme :code:`furo`, visit
`sphinx documentation <https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html>`_ &
`furo documentation <https://sphinx-themes.org/sample-sites/furo/>`_ .

To use `doctest`:

- Use :code:`make doctest` in :code:`toqito/docs` for the docstring examples to be verified. 
- Use :code:`pytest  --doctest-glob=*.rst` to check the examples in all the :code:`.rst` files in :code:`toqito/docs` work as expected. If
  you would like to only check the examples in a  specific file, use :code:`pytest  --doctest-glob=tutorials.name_of_file.rst`
  instead. 

--------------------
Adding a new feature
--------------------


If you add a new feature to :code:`toqito`, make sure

- The function docstring follows the style guidelines as specified in `References in Docstrings`_.
- Added lines should show up as covered in the :code:`pytest` code coverage report. See `Testing`_.
- Code and tests for the new feature should follow the style guidelines as discussed in `Code Style`_.
- Finally, if the new feature is a new module, it has to be listed in :code:`docs/autoapi_members.rst` such that the new module appears
  in the :code:`API Reference` page due to :code:`sphinx-autoapi`.


---------------------
Additional Resources
---------------------

-    `General GitHub documentation <https://help.github.com/>`_
-    `PR best practices <http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/>`_
-    `A guide to contributing to software packages <http://www.contribution-guide.org>`_
