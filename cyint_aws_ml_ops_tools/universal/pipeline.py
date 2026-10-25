import os

from botocore.exceptions import ClientError


def prepare_pipeline_variables(
    environment_prefix=None, aws_access_key=None, aws_secret_key=None, region=None
):
    """
    Creates the necessary pipeline specific variables based on environment variables or
    overidden values
    """

    environment_prefix_name = environment_prefix

    aws_access_key_id = (
        os.environ["AWS_ACCESS_KEY"] if aws_access_key is None else aws_access_key
    )
    aws_secret_key_id = (
        os.environ["AWS_SECRET_KEY"] if aws_secret_key is None else aws_secret_key
    )
    region_name = os.environ["AWS_REGION"] if region is None else region

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
