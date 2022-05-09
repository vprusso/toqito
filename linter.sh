#!/bin/bash          
echo "Running tests..."
pytest --cov-report term-missing --cov=toqito tests
echo "Running black on toqito/..."
black toqito
echo "Running black on tests/..."
black tests
echo "Running pydocstyle on toqito/..."
pydocstyle toqito
echo "Running pydocstyle on tests/..."
pydocstyle tests
echo "Running pylint on toqito/..."
pylint toqito
echo "Running pylint on tests/"
pylint tests
echo "Rebuilding docs..."
make clean html -C docs
echo "Checking line count..."
wc -l **/*.py
