import os

import boto3
from botocore.exceptions import ClientError


def define_feature_group(
    name,
    description,
    record_identifier_feature,
    event_time_feature,
    feature_definitions,
    online_store_config,
    offline_store_config,
    s3_key=None,
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    role=None,
    data_lake_bucket=None,
    environment_prefix=None,
):
    """
    Primary function to call when defining a new feature group in a Jupyter notebook or script. This function is designed to succeed regardless of
    existing state of the system.
    """
    environment_prefix_name = environment_prefix

    aws_access_key_id = (
        os.environ["AWS_ACCESS_KEY"] if aws_access_key is None else aws_access_key
    )
    aws_secret_key_id = (
        os.environ["AWS_SECRET_KEY"] if aws_secret_key is None else aws_secret_key
    )
    region_name = os.environ["AWS_REGION"] if region is None else region
    role_arn = os.environ["SAGEMAKER_ROLE"] if role is None else role

    environment_prefix_name = (
        os.environ["ENVIRONMENT_PREFIX"]
        if environment_prefix is None
        else environment_prefix
    )

    sagemakerclient = boto3.client(
        "sagemaker",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    feature_group_name = f"{environment_prefix_name}{name}"
    feature_group_name = feature_group_name.replace("_", "-")

    is_different = False
    is_new = is_feature_group_new(sagemakerclient, feature_group_name)
    if not is_new:
        is_different = is_feature_group_different(
            sagemakerclient,
            feature_group_name,
            record_identifier_feature,
            event_time_feature,
            feature_definitions,
            online_store_config,
            offline_store_config,
            role_arn,
            description,
        )

    if is_different:
        sagemakerclient.delete_feature_group(FeatureGroupName=feature_group_name)

    response = sagemakerclient.create_feature_group(
        FeatureGroupName=feature_group_name,
        RecordIdentifierFeatureName=record_identifier_feature,
        EventTimeFeatureName=event_time_feature,
        FeatureDefinitions=feature_definitions,
        OnlineStoreConfig=online_store_config,
        OfflineStoreConfig=offline_store_config,
        RoleArn=role_arn,
        Description=description,
    )

    return {"feature_group": response}


def is_feature_group_new(sagemakerclient, feature_group_name):
    try:
        response = sagemakerclient.describe_feature_group(
            FeatureGroupName=feature_group_name
        )
        return False
    except ClientError as e:
        return True


def is_feature_group_different(
    sagemakerclient,
    feature_group_name,
    record_identifier_feature,
    event_time_feature,
    feature_definitions,
    online_store_config,
    offline_store_config,
    role_arn,
    description,
):

    response = sagemakerclient.describe_feature_group(
        FeatureGroupName=feature_group_name
    )

    return (
        response["RoleArn"] != role_arn
        or response["RecordIdentifierFeatureName"] != record_identifier_feature
        or response["EventTimeFeatureName"] != event_time_feature
        or response["FeatureDefinitions"] != feature_definitions
        or response["FeatureDefinitions"] != feature_definitions
        or response["OnlineStoreConfig"] != online_store_config
        or response["OfflineStoreCOnfig"] != offline_store_config
        or response["Description"] != description
    )
