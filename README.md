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
Don't forget to increment the version number in `pyproject.toml` prior to building.
`poetry build` to create the `dist` folder containing the package build.

## Deploy to pypi

Increment the version number `pyproject.toml`.

Tag the release.

run `python3 -m twine upload ./dist/*` to upload to pypi. Currently this package is deployed to the `cyint` pypi account.

You might run into the following error:

    HTTPError: 400 Client Error: File already exists. See https://pypi.org/help/#file-name-reuse for url: https://upload.pypi.org/legacy/

If that happens to you, check up on 2 things:

* Make sure you updated the version number in both files
* Delete the old version files from your dist/ directory


## Version 2 security migration (candidate)

Version 2 requires Python 3.12.15 or a newer security release in the 3.12 series.
Runtime dependencies are pinned in pyproject.toml and poetry.lock. The unused
smdebug and AWS CLI runtime dependencies were removed; uuid4 uses Python's
standard library. Consumers that need CLI/debug tooling must declare it separately.

The library buildspec runs offline unit tests, never a notebook or production
pipeline. It does not request environment secrets or write a .env file. Tests
replace cloud credentials with dummy values and deny network and subprocess calls.
This boundary is test-only; no production security agent or firewall is changed.

A changed feature-group definition now raises before deletion unless the caller
passes allow_recreate=True explicitly after reviewing data retention and recovery.
The flag is keyword-only and accepts a real boolean, not a string or integer.
Creating a new group and reading an existing matching group retain their prior API.

All three downstream consumers must be qualified against the exact candidate
artifact before release. The missing historical 1.4 wheel is not silently replaced
by a public package or treated as verified. No deployment follows from unit tests.

Permission, throttling and validation errors during feature-group/job/trigger
existence checks now propagate instead of being treated as evidence that a
resource is missing. Only explicit provider not-found codes mean missing.

Experiments no longer depend on the retired smexperiments package or pkg_resources.
The library uses the current boto3 SageMaker experiment/trial/component APIs.
Its parameter-only tracker records start/completion/failure and associates metadata
components with each training trial. Scalar strings/numbers are retained; grids
and structured configuration use JSON strings. Exception details are not copied
into cloud metadata. Artifact uploads remain explicit application operations.
Bucket creation occurs once, uses the correct regional location constraint, and
only an already-owned-bucket response is accepted as a successful existing bucket.
