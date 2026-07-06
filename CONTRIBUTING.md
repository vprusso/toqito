# Contributing to `toqito`

Thanks for your interest in contributing to `|toqito⟩`! Contributions of every
size are welcome — fixing a typo, sharpening an error message, adding a missing
docstring example, writing a test, or implementing a new quantum-information
routine.

This file is a quick entry point. The **full, authoritative guide** lives in the
documentation:

- 📖 **[Contributing Guide](https://vprusso.github.io/toqito/contributing-guide/)**
  — dev setup, testing, code style, docstring/reference conventions, and how to
  add a new feature.

## Good first issues

If you are looking for a small, self-contained place to start, browse the
[**`good first issue`** queue](https://github.com/vprusso/toqito/labels/good%20first%20issue).
These are scoped so you can finish them without deep knowledge of the codebase —
adding an example, improving a docstring, tightening an error message, covering
an edge case with a test, or benchmarking a function.

Issues are labelled by **type** (`bug`, `documentation`, `enhancement`,
`refactor`, …) and by **area** (`examples`, `gallery`, `benchmarking`, …), so you
can filter to what interests you. If something is unclear, comment on the issue —
we are happy to help you get started.

## Quick start

```bash
# Fork and clone your fork, then from the repository root:
uv sync                 # create the environment and install toqito (editable)

uv run pytest           # run the test suite
uv run ruff check       # lint
uv run ruff format      # auto-format
```

`toqito` targets **Python >= 3.12** and uses [`uv`](https://docs.astral.sh/uv/)
for environment management.

## Making a change

1. Create a branch off `master`.
2. Make your change with a matching test. Code lives in `toqito/<module>/` and
   its tests in `toqito/<module>/tests/`. See the
   [repository layout](https://vprusso.github.io/toqito/contributing-guide/#repository-layout).
3. New public functions need a docstring with a theoretical description, an
   `Examples` section using a `markdown-exec` fenced block, and a `References`
   section, and must be exported from the module's `__init__.py`. See
   [Adding a new feature](https://vprusso.github.io/toqito/contributing-guide/#adding-a-new-feature).
4. Before opening the PR, confirm locally:
   - [ ] `uv run ruff check` and `uv run ruff format --check` pass.
   - [ ] `uv run pytest` passes, and new lines are covered.
   - [ ] The docs build cleanly (`uv sync --group docs && uv run mkdocs build -f docs/mkdocs.yml`).
5. Open the pull request and fill in the template. Reference any issue it closes
   with `Closes #<number>`.

## Reporting issues

Use the [issue templates](https://github.com/vprusso/toqito/issues/new/choose):
bug report, new function request, documentation improvement, mathematical /
reference correction, or performance improvement.

## Code of conduct and licensing

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE). Please keep interactions respectful and
constructive.
