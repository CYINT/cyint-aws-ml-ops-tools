# cyint-aws-ml-ops-tools

A python package that wraps some useful functions to be used in the modern ML Ops pipelines.

[![codecov](https://codecov.io/gh/CYINT/cyint-aws-ml-ops-tools/branch/main/graph/badge.svg?token=2VLUo3hBph)](https://codecov.io/gh/CYINT/cyint-aws-ml-ops-tools)

## Dependencies

Setup a virtual environment, then install `poetry` using `pip`

`pip install poetry`

Now use poetry to install dependencies

`poetry install`

And you can enter the poetry virtual environment as follows:

`poetry shell`

From here, you can set pre-commit hooks to ensure proper commit formatting:

`pre-commit install`

## Build

Delete the `dist` folder if it already exists.
Don't forget to increment the version number in `setup.py `prior to building.
`poetry build` to create the `dist` folder containing the package build.

## Deploy to pypi

Increment the version number `setup.py`.

Tag the release.

run `python3 -m twine upload ./dist/*` to upload to pypi. Currently this package is deployed to the `cyint` pypi account.

You might run into the following error:

    HTTPError: 400 Client Error: File already exists. See https://pypi.org/help/#file-name-reuse for url: https://upload.pypi.org/legacy/

If that happens to you, check up on 2 things:

* Make sure you updated the version number in both files
* Delete the old version files from your dist/ directory
