import os
import math
import re

from botocore.exceptions import ClientError
from botocore.validate import validate_parameters



def resolve_aws_credentials(aws_access_key=None, aws_secret_key=None):
    """Keep complete explicit/legacy pairs; otherwise use the SDK provider chain.

    Incomplete pairs are errors, never silently combined across credential sources.
    Return None values so boto3 can use standard environment/session-token, role,
    web-identity or configured provider credentials without generating a .env file.
    """
    if aws_access_key is not None or aws_secret_key is not None:
        key, secret = aws_access_key, aws_secret_key
    else:
        key, secret = os.environ.get("AWS_ACCESS_KEY"), os.environ.get("AWS_SECRET_KEY")
    if key is None and secret is None:
        return None, None
    if not isinstance(key, str) or not key or not isinstance(secret, str) or not secret:
        raise ValueError("AWS credentials require a complete nonempty key/secret pair")
    return key, secret


def prepare_pipeline_variables(
    environment_prefix=None, aws_access_key=None, aws_secret_key=None, region=None
):
    """
    Creates the necessary pipeline specific variables based on environment variables or
    overidden values
    """

    environment_prefix_name = environment_prefix

    aws_access_key_id, aws_secret_key_id = resolve_aws_credentials(aws_access_key, aws_secret_key)
    region_name = region if region is not None else os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))

    environment_prefix_name = (
        os.environ["ENVIRONMENT_PREFIX"]
        if environment_prefix is None
        else environment_prefix
    )

    return [environment_prefix_name, aws_access_key_id, aws_secret_key_id, region_name]


def create_s3bucket_if_not_exist(bucket_name, s3client):
    request = {"Bucket": bucket_name}
    region = s3client.meta.region_name
    if region and region != "us-east-1":
        request["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        s3client.create_bucket(**request)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "BucketAlreadyOwnedByYou":
            raise


def sanitize_bucket_name(bucket_name):
    return bucket_name.replace("_", "-")


def validate_request_model(shape, request):
    """Validate structure plus declared limits/enums/patterns before any writes."""
    validate_parameters(request, shape)
    validate_model_constraints(shape, request)


def validate_model_constraints(shape, value):
    kind = shape.type_name
    measured = None
    if kind in {"string", "list", "map", "blob"} and hasattr(value, "__len__"):
        measured = len(value)
    elif kind in {"integer", "long", "float", "double"}:
        if isinstance(value, bool):
            raise ValueError("Boolean is not a numeric service-model value")
        if kind in {"integer", "long"}:
            bits = 32 if kind == "integer" else 64
            if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
                raise ValueError("Numeric value exceeds the service-model integer range")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError("Numeric service-model values must be finite")
        measured = value
    if measured is not None:
        minimum, maximum = shape.metadata.get("min"), shape.metadata.get("max")
        if minimum is not None and measured < minimum:
            raise ValueError("Request value is below its declared service-model minimum")
        if maximum is not None and measured > maximum:
            raise ValueError("Request value exceeds its declared service-model maximum")
    if kind == "structure":
        if shape.metadata.get("union") and len(value) != 1:
            raise ValueError("Service-model union requires exactly one member")
        for key, item in value.items():
            validate_model_constraints(shape.members[key], item)
    elif kind == "list":
        for item in value:
            validate_model_constraints(shape.member, item)
    elif kind == "map":
        for key, item in value.items():
            validate_model_constraints(shape.key, key)
            validate_model_constraints(shape.value, item)
    elif kind == "string":
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError:
            raise ValueError("Service-model strings must be valid UTF-8") from None
        if shape.enum and value not in shape.enum:
            raise ValueError("Request contains an unsupported service-model enumerated value")
        pattern = shape.metadata.get("pattern")
        if pattern:
            # AWS Glue expresses supplementary Unicode as a Java UTF-16 pair range.
            # Python regexes use Unicode scalars; preserve that range rather than
            # skipping the provider pattern or rejecting valid supplementary text.
            pattern = pattern.replace(r"\uD800\uDC00-\uDBFF\uDFFF", r"\U00010000-\U0010FFFF")
            pair_range = chr(0xD800) + chr(0xDC00) + "-" + chr(0xDBFF) + chr(0xDFFF)
            pattern = pattern.replace(pair_range, chr(0x10000) + "-" + chr(0x10FFFF))
            try:
                matched = re.fullmatch(pattern, value)
            except re.error as error:
                raise ValueError("Unsupported service-model regex; request not dispatched") from error
            if matched is None:
                raise ValueError("Request does not match its service-model string pattern")
