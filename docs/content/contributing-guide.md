# Contributing Guide

We welcome contributions from external contributors, and this document
describes how to merge code changes into `|toqito⟩`.

1.  Make sure you have a [GitHub
    account](https://github.com/signup/free).
2.  [Fork](https://help.github.com/articles/fork-a-repo/) this
    repository on GitHub.
3.  On your local machine,
    [clone](https://help.github.com/articles/cloning-a-repository/) your
    fork of the repository. You will have to install an editable version
    on your local machine. Instructions are provided below.

!!! warning
    Avoid ad-hoc `pip install -e .` workflows; the project standardizes on
    `uv` for syncing dependencies.

4.  As stated in [Getting started](./getting-started.md), ensure you have Python 3.11 or greater installed on
    your machine or in a virtual environment
    ([pyenv](https://github.com/pyenv/pyenv), [pyenv
    tutorial](https://realpython.com/intro-to-pyenv/)). Consider using a
    [virtual environment](https://docs.python.org/3/tutorial/venv.html).
    You can also use `pyenv` with `virtualenv` [to manage different
    Python versions](https://github.com/pyenv/pyenv-virtualenv) or
    `conda` to create virtual environments with [different Python
    versions](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments).
5.  You will require [uv](https://docs.astral.sh/uv/) to manage the
    dependencies of `toqito`. Refer to the [uv installation
    guide](https://docs.astral.sh/uv/getting-started/installation/) for
    platform-specific instructions.
6.  Now, navigate to your local clone of the `|toqito⟩` repository as
    shown below.

``` bash
$ cd toqito/
```

7.  Use `uv` as shown below in the `|toqito⟩` folder. This installs an
    editable version of `|toqito⟩` along with the default development
    tools.

``` bash
toqito/ $ uv sync
```

You are now free to make the desired changes in your fork of `|toqito⟩`.

## Making Changes

1.  Add some really awesome code to your local fork. It's usually a
    [good
    idea](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/)
    to make changes on a
    [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
    with the branch name relating to the feature you are going to add.
2.  When you are ready for others to examine and comment on your new
    feature, navigate to your fork of `|toqito⟩` on GitHub and open a
    [pull
    request(PR)](https://help.github.com/articles/using-pull-requests/)
    . Note that after you launch a PR from one of your fork's branches,
    all subsequent commits to that branch will be added to the open pull
    request automatically. Each commit added to the PR will be validated
    for mergeability, compilation and test suite compliance; the results
    of these tests will be visible on the PR page.
3.  If you're adding a new feature, you must add test cases and
    documentation. See [Adding a new feature](#adding-a-new-feature) for
    a detailed checklist.
4.  When the code is ready to go, make sure you run the test suite using
    `pytest`, `ruff`, etc.
5.  When you're ready to be considered for merging, comment on your PR
    that it is ready for a review to let the `|toqito⟩` devs know that
    the changes are complete. The code will not be reviewed until you
    have commented so, the continuous integration workflow passes, and
    the primary developer approves the reviews.

## Adding a new feature

If you add a new feature to `|toqito⟩`, make sure

-   The function docstring follows the style guidelines as specified in
    [References in Docstrings](#references-in-docstrings).
-   The docstring of a new feature should contain a theoretical
    description of the feature, one or more examples in an `Examples`
    subsection and a `References` subsection. The docstring code
    examples should utilize fenced code blocks with `markdown-exec`.
-   Added lines should show up as covered in the `pytest` code coverage
    report. See [Testing](#testing).
-   Code and unit tests for the new feature should follow the style
    guidelines as discussed in [Code Style](#code-style)
-   The new feature must be added to the `init` file of its module to
    avoid import issues.

## Testing

A convenient way to verify if the installation procedure worked
correctly, use [pytest]{.title-ref} in the `|toqito⟩` folder as shown
below.

``` bash
toqito/ $ uv run pytest
```

The `pytest` module is used for testing and `pytest-cov` can be used to
generate coverage reports locally. In order to run and `pytest`, you
will need to ensure it is installed on your machine along with
`pytest-cov`. If the editable installation process worked without any
issues, both `pytest` and `pytest-cov` should be installed in your local
environment.

If not, consult the [pytest](https://docs.pytest.org/en/latest/) and
[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) websites for
additional options on code coverage reports. For example, if your
addition is not properly covered by tests, code coverage can be checked
by using `--cov-report term-missing` options in `pytest-cov`.

If you are making changes to `toqito.some_module`, the corresponding
tests should be in `toqito/some_module/tests`.

A beginner introduction to adding unit tests is available
[here](https://third-bit.com/py-rse/testing.html) .

!!! note
    Performance benchmarks are not part of the standard test run. Trigger
    the `Benchmark Regression Analysis` GitHub workflow manually when you
    need timings or regression checks.

## Code Style 

We use `ruff` to check for formatting issues. Consult the documentation
for [ruff](https://docs.astral.sh/ruff/tutorial/#getting-started) for
additional information.

Do not use an autoformatter like `black` as the configuration settings
for `ruff` as specified in
[pyproject.toml](https://github.com/vprusso/toqito/blob/master/pyproject.toml)
might be incompatible with the changes made by `black`. This is
discussed in detail at [this
link](https://docs.astral.sh/ruff/formatter/black/).

Static typing is enforced with `mypy` (see [mypy
documentation](https://mypy.readthedocs.io/en/stable/)). Before
submitting a pull request, run the type checker against the source tree
(the type checker lives in the `lint` dependency group):

``` bash
uv run --group lint mypy toqito
```

### Setting Up Pre-Commit Hooks

Pre-commit hooks ensure that the code meets our formatting and linting
standards before it is committed to the repository. Install the hooks
with the following command.

``` bash
uv run pre-commit install
```

This integrates ruff checks into your workflow, ensuring consistent code
quality across the project.

Additionally, the commit-msg hook ensures adherence to the [Conventional
Commits](https://www.conventionalcommits.org/) format for all commit
messages and helps maintain a standardized commit history.

``` bash
uv run pre-commit install --hook-type commit-msg
```

## References in Docstrings

If you are adding a new function, make sure the docstring of your
function follows the formatting specifications in [Code Style](#code-style). A standard format for `|toqito⟩` docstring is provided below:

``` python
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
```

Use `\(\)` for inline math and `\[\]` for display math in docstrings.
Use `[@ citation_key]` for citations in docstrings (e.g., `[@johnston2014counting]`).

To add an attribution to a paper or a book, add your reference with
`some_ref` as the citation key to `docs/content/refs.bib`. All references in
`refs.bib` are arranged alphabetically according to the first author's
last name. Take a look at the [existing
entries](https://github.com/vprusso/toqito/blob/master/docs/content/refs.bib) to
get an idea of how to format the `bib` keys.
## Documentation 

We use `sphinx` to build the documentation. Sync the docs dependency
group first (`uv sync --group docs`), then run:

``` bash
toqito/docs$ uv run make clean html
```

If you would prefer to decrease the amount of time taken by `sphinx` to
build the documentation locally, use `make html` instead after the
documentation has been built once.

A standard document has to follow the `.rst` format. For more
information on `sphinx`, `rst` fromat and the documentation theme
`furo`, visit [sphinx
documentation](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html)
, [rst
primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
& [furo documentation](https://sphinx-themes.org/sample-sites/furo/) .

## Additional Resources

-   [General GitHub documentation](https://help.github.com/)
-   [PR best
    practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
-   [A guide to contributing to software
    packages](http://www.contribution-guide.org)
