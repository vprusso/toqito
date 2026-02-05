.. _contrib_guide_reference-label:

""""""""""""""""""
Contributing Guide
""""""""""""""""""

We welcome contributions from external contributors, and this document describes how to merge code changes into
:code:`|toqito⟩`. 


1. Make sure you have a `GitHub account <https://github.com/signup/free>`_.
2. `Fork <https://help.github.com/articles/fork-a-repo/>`_ this repository on GitHub.
3. On your local machine, `clone <https://help.github.com/articles/cloning-a-repository/>`_ your fork of the repository. You will
   have to install an editable version on your local machine. Instructions are provided below.


.. warning::
     Avoid ad-hoc :code:`pip install -e .` workflows; the project standardizes on :code:`uv` for syncing dependencies.

4. As stated in :ref:`getting_started_reference-label`, ensure you have Python 3.10 or greater installed on your machine or in
   a virtual environment (`pyenv <https://github.com/pyenv/pyenv>`_, `pyenv tutorial <https://realpython.com/intro-to-pyenv/>`_).
   Consider using a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.
   You can also use :code:`pyenv` with :code:`virtualenv` `to manage different Python
   versions <https://github.com/pyenv/pyenv-virtualenv>`_ or :code:`conda` to create virtual environments with `different Python
   versions <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments>`_.

5. You will require `uv <https://docs.astral.sh/uv/>`_ to manage the dependencies of :code:`toqito`.  
   Refer to the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_ for
   platform-specific instructions.


6. Now, navigate to your local clone of the :code:`|toqito⟩` repository as shown below.

.. code-block:: bash

    $ cd toqito/

7. Use :code:`uv` as shown below in the :code:`|toqito⟩` folder. This installs an editable version of :code:`|toqito⟩`
   along with the default development tools.

.. code-block:: bash

    toqito/ $ uv sync

You are now free to make the desired changes in your fork of :code:`|toqito⟩`. 

--------------
Making Changes
--------------

1.   Add some really awesome code to your local fork.  It's usually a 
     `good idea <http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/>`_
     to make changes on a 
     `branch <https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/>`_
     with the branch name relating to the feature you are going to add.

2.   When you are ready for others to examine and comment on your new feature,
     navigate to your fork of :code:`|toqito⟩` on GitHub and open a 
     `pull request(PR) <https://help.github.com/articles/using-pull-requests/>`_ . Note that
     after you launch a PR from one of your fork's branches, all subsequent commits to that branch will be added to the
     open pull request automatically.  Each commit added to the PR will be validated for mergeability, compilation and
     test suite compliance; the results of these tests will be visible on the PR page.

3.   If you're adding a new feature, you must add test cases and documentation. See `Adding a new feature`_
     for a detailed checklist. 

4.   When the code is ready to go, make sure you run the test suite using :code:`pytest`, :code:`ruff`, etc.

5.   When you're ready to be considered for merging, comment on your PR that it is ready for a review
     to let the :code:`|toqito⟩` devs know that the changes are complete. The code will not be reviewed
     until you have commented so, the continuous integration workflow passes, and the primary developer approves the
     reviews.

--------------------
Adding a new feature
--------------------


If you add a new feature to :code:`|toqito⟩`, make sure

- The function docstring follows the style guidelines as specified in `References in Docstrings`_. 
- The docstring of a new feature should contain a theoretical description of the feature, one or more examples in an :code:`Examples`
  subsection and a :code:`References` subsection. The docstring code examples should utilize `jupyter-sphinx <https://jupyter-sphinx.readthedocs.io/en/latest/>`_. 
- Added lines should show up as covered in the :code:`pytest` code coverage report. See `Testing`_.
- Code and unit tests for the new feature should follow the style guidelines as discussed in :ref:`Code Style <code_style_reference-label>`.
- The new feature must be added to the :code:`init` file of its module to avoid import issues. 
- Finally, if the new feature is a new module, it has to be listed in :code:`docs/autoapi_members.rst` such that the new module appears
  in the :code:`API Reference` page due to :code:`sphinx-autoapi`.

-------
Testing
-------

A convenient way to verify if the installation procedure worked correctly, use `pytest` in the :code:`|toqito⟩` folder as
shown below.

.. code-block:: bash

    toqito/ $ uv run pytest

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

A beginner introduction to adding unit tests is available `here <https://third-bit.com/py-rse/testing.html>`_ .

.. note::
    Performance benchmarks are not part of the standard test run. Trigger the
    ``Benchmark Regression Analysis`` GitHub workflow manually when you need
    timings or regression checks.


.. _code_style_reference-label:

----------
Code Style
----------


We use :code:`ruff` to check for formatting issues. Consult the documentation for
`ruff <https://docs.astral.sh/ruff/tutorial/#getting-started>`_ for additional information.

Do not use an autoformatter like :code:`black` as the configuration settings for :code:`ruff` as specified in
`pyproject.toml <https://github.com/vprusso/toqito/blob/master/pyproject.toml>`_
might be incompatible with the changes made by :code:`black`. This is discussed in detail at
`this link <https://docs.astral.sh/ruff/formatter/black/>`_.

Static typing is enforced with :code:`mypy` (see `mypy documentation <https://mypy.readthedocs.io/en/stable/>`_). Before submitting a pull request, run the
type checker against the source tree (the type checker lives in the ``lint`` dependency group):

.. code-block:: bash

    uv run --group lint mypy toqito

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Setting Up Pre-Commit Hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pre-commit hooks ensure that the code meets our formatting and linting standards before it is committed to the repository. Install the hooks with the following command.

.. code-block:: bash
   
   uv run pre-commit install

This integrates ruff checks into your workflow, ensuring consistent code quality across the project. 

Additionally, the commit-msg hook ensures adherence to the `Conventional Commits <https://www.conventionalcommits.org/>`_ format for all commit messages and helps maintain a standardized commit history.

.. code-block:: bash

    uv run pre-commit install --hook-type commit-msg

------------------------
References in Docstrings
------------------------


If you are adding a new function, make sure the docstring of your function follows the formatting specifications
in :ref:`Code Style <code_style_reference-label>`. A standard format for :code:`|toqito⟩` docstring is provided below:

.. code-block:: python
    
    def my_new_function(some_parameter: parameter_type) -> return_type:
        r"""One liner description of the new function.

            Detailed description of the function.

            Examples
            ==========
            Demonstrate how the function works with expected output.

            .. jupyter-execute::

                import numpy as np
                x = np.array([[1, 2], [3, 4]])
                print(x)

            References
            ==========
            .. footbibliography::
                
        
            :param name_of_parameter: Description of the parameter.
            :raises SomeError: Description for when the function raises an error.
            :return: Description of what the function returns.
                
        """

Use :code:`.. math::` mode for equations and use use :code:`:cite:some_ref` for some reference in the docstring. 

To add an attribution to a paper or a book, add your reference with :code:`some_ref` as the citation key to 
``docs/refs.bib``. All references in ``refs.bib`` are arranged alphabetically according to the first author's last name. Take a
look at the `existing entries <https://github.com/vprusso/toqito/blob/master/docs/refs.bib>`_ to get an idea of how to format the ``bib`` keys. 

Following is used in a docstring for the references to show up in the documentation build.

.. code-block:: text

    References
    ==========
    .. footbibliography::
        


--------------
Documentation
--------------


We use :code:`sphinx` to build the documentation. Sync the docs dependency group first (``uv sync --group docs``),
then run:

.. code-block:: bash

    toqito/docs$ uv run make clean html

If you would prefer to decrease the amount of time taken by :code:`sphinx` to build the documentation locally, use :code:`make html`
instead after the documentation has been built once.

A standard document has to follow the :code:`.rst` format.  For more information on :code:`sphinx`, :code:`rst` fromat and
the documentation theme :code:`furo`, visit
`sphinx documentation <https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html>`_ , 
`rst primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ &
`furo documentation <https://sphinx-themes.org/sample-sites/furo/>`_ .

---------------------
Additional Resources
---------------------

-    `General GitHub documentation <https://help.github.com/>`_
-    `PR best practices <http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/>`_
-    `A guide to contributing to software packages <http://www.contribution-guide.org>`_
