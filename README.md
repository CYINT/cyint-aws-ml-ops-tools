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


Glue script upload and feature-group bucket preparation now use the same regional,
fail-closed helper. Script upload completes before job/trigger mutation. A bucket
failure stops feature-group deletion; explicit recreation polls for provider
not-found for up to 120 checks separated by 10 seconds, and stops on denial,
deletion failure or timeout. API latency and retries add to that polling interval.
This is not a backup or a recovery guarantee; approved recreation still deletes
online records. Existing offline records and catalog lifecycle remain AWS behavior.

Winner selection maximizes ordinary scores and minimizes error metrics. A failure
reporting tracking completion propagates; if the tracked body also failed, the
original body exception remains primary with a generic completion-unconfirmed note.

Complete explicit credentials and the legacy AWS_ACCESS_KEY/AWS_SECRET_KEY pair
remain supported. If neither is supplied, boto3 uses its normal credential provider
chain, including session tokens and execution roles. Partial/empty pairs are rejected
without mixing credential sources. Do not put credentials in notebook or .env output.

The offline test audit hook denies socket/DNS, subprocess and shell execution using
PermissionError. Standard-library metadata probes can handle this denial using native
fallbacks; no launch is permitted and no platform or SDK prewarming is required.


Glue trigger requests omit inapplicable Schedule/Predicate/EventBatchingCondition
fields, preserve custom actions and validate type intent before script/job mutation.
Existing trigger types must match; changing type requires an explicit migration.
Caller configuration dictionaries are not modified. Real Glue service-model tests
cover new/existing ON_DEMAND, SCHEDULED, CONDITIONAL and EVENT requests.

Model promotion enumerates every trial-component page in each polling pass. It
rejects failed/stopped/unknown states, duplicate training components, nonterminating
pagination, ambiguous/nonfinite metrics and unconfirmed detailed completion before
any staging write. Running/stopping trials are polled for up to max_wait_attempts
(default 3600), with wait_interval seconds between checks (default 1); API latency
and retries are additional. Both settings are keyword-only on promote_winner and
detect_experiment_winner. A timeout does not stop the training job or stage a model.
All candidate artifacts remain subject to consumer, Linux and deployment acceptance.


The public Glue job entrypoint validates complete create/update job and trigger
payloads against the installed botocore request schemas, then reads current job and
trigger state and checks trigger compatibility before its first S3/job/trigger write.
Unknown or denied state lookups and structurally invalid nested configuration stop
before mutation. Job configuration is copied rather than modified in place.
This preflight does not make multi-service operations atomic: later provider I/O
failures can leave partial changes, so callers must reconcile state before retrying.


Feature-group creation/replacement prepares an independent deep copy and validates
its complete SDK request before clients, bucket creation or deletion. Enum/pattern,
feature-name/identifier/event-time and known store-configuration constraints are
checked locally. Online-store defaults remain enabled; an empty offline config is
omitted. Neither input dictionaries nor shared defaults are changed. Prefix overrides
retain an explicitly supplied offline KMS key. Comparison reads every feature page,
rejects incomplete/unstable enumeration, and normalizes provider defaults on copies
without injecting resolved S3/catalog fields into the create request. An unchanged
group returns without bucket or feature-group writes. Explicit recreation still
requires reviewed data retention/recovery and confirmed absence; a later provider
I/O failure can leave the old online store deleted. These checks are not a restore.

The public Glue job binds ScriptLocation to the same bucket/key uploaded by the
library. destination_key accepts a raw object key or an S3 URI in the configured
script bucket; cross-bucket URIs are rejected before any write. Raw keys are retained
for upload and encoded for the canonical script URI. S3 URI paths are decoded once;
queries, fragments and invalid encodings are rejected rather than silently remapped.
Direct handle_job callers continue to supply their intended existing script location.


Existing feature groups must be confirmed Created before ordinary reconciliation.
Creating, deleting, failed or unknown lifecycle states require separate recovery or
completion handling; allow_recreate does not bypass that readiness gate. New create
responses remain asynchronous provider acknowledgements, not readiness/recovery proof.
Shared request validation enforces declared min/max, numeric ranges/finite values,
UTF-8, enums and patterns for Glue and feature requests, including the AWS Glue
UTF-16 supplementary-character range translated to Python Unicode scalars. Known
collection rules preserve valid InMemory List/Set and Fractional Vector definitions
while rejecting unsupported tiers, invalid vector type/config/dimensions and multiple
vectors. Iceberg event time must be String and Glue table creation must remain enabled.

A missing or empty hyperparameter grid creates one static-only parameter combination
per dataset. Static parameters, experiment/trial/component metadata and the training
callback are preserved; a nonempty grid still produces its Cartesian combinations.

S3 script URIs reject literal ASCII control characters before URI parsing. Raw
object keys preserve their bytes, including control characters; percent-encoded
URI keys are decoded once and encoded in the canonical ScriptLocation without
silently stripping tabs or line breaks.
