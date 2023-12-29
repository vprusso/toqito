.. _contrib_guide_reference-label:

""""""""""""""""""
Contributing Guide
""""""""""""""""""

We welcome contributions from external contributors, and this document
describes how to merge code changes into this `toqito`. 

================
Getting Started
================

-    Make sure you have a [GitHub account](https://github.com/signup/free).
-    [Fork](https://help.github.com/articles/fork-a-repo/) this repository on GitHub.
-    On your local machine,
     [clone](https://help.github.com/articles/cloning-a-repository/) your fork of
     the repository.
-    To install an editable version on your local machine, run `pip install -e .` in
     the top-level directory of the cloned repository.

-------
Testing
-------

The :code:`pytest` module is used for testing and :code:`pytest-cov` is used to generate
coverage reports. In order to run and :code:`pytest`, you will need to ensure it is
installed on your machine along with :code:`pytest-cov`. Consult the `pytest <https://docs.pytest.org/en/latest/>`_ 
and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ websites for more
information. To run the suite of tests for :code:`toqito`, run the following command
in the root directory of this project:

.. code-block:: bash

    pytest --cov-report term-missing --cov=toqito tests/


==============
Making Changes
==============

-    Add some really awesome code to your local fork.  It's usually a 
     [good idea](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/)
     to make changes on a 
     [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
     with the branch name relating to the feature you are going to add.
-    When you are ready for others to examine and comment on your new feature,
     navigate to your fork of `toqito` on GitHub and open a 
     [pull request](https://help.github.com/articles/using-pull-requests/) (PR). Note that
     after you launch a PR from one of your fork's branches, all
     subsequent commits to that branch will be added to the open pull request
     automatically.  Each commit added to the PR will be validated for
     mergability, compilation and test suite compliance; the results of these tests
     will be visible on the PR page.
-    If you're providing a new feature, you must add test cases and documentation. We use `sphinx`
     to build the documentation. To build the documentation locally via `make html` in the
     `toqito/docs` directory, make sure `sphinx` and `sphinx-rtd-theme` are installed.
     For more information, visit [sphinx documentation](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html) & [sphinx-rtd-theme documentation](https://sphinx-rtd-theme.readthedocs.io/en/stable/installing.html)
-    When the code is ready to go, make sure you run the test suite using pytest.
-    When you're ready to be considered for merging, check the "Ready to go"
     box on the PR page to let the `toqito` devs know that the changes are complete.
     The code will not be merged until this box is checked, the continuous
     integration returns check marks,
     and the primary developer approves the reviews.

---------------------
Additional Resources
---------------------

-    [General GitHub documentation](https://help.github.com/)
-    [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
-    [A guide to contributing to software packages](http://www.contribution-guide.org)
-    [Thoughtful PR example](http://www.thinkful.com/learn/github-pull-request-tutorial/#Time-to-Submit-Your-First-PR)

