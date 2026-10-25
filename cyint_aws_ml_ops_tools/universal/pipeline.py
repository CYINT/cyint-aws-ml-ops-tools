import os

from botocore.exceptions import ClientError



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
