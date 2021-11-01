import os
from time import sleep

import boto3
from botocore.exceptions import ClientError
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session


def define_feature_group(
    name,
    description,
    record_identifier_feature,
    event_time_feature,
    feature_definitions,
    online_store_config={"EnableOnlineStore": True},
    offline_store_config={},
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

    environment_prefix_name = (
        os.environ["ENVIRONMENT_PREFIX"]
        if environment_prefix is None
        else environment_prefix
    )

    data_lake_bucket_name = (
        os.environ["DATA_LAKE_BUCKET"] if data_lake_bucket is None else data_lake_bucket
    )
    data_lake_bucket_name = f"{environment_prefix_name}{data_lake_bucket_name}"
    data_lake_bucket_name = data_lake_bucket_name.replace("_", "-")

    region_name = os.environ["AWS_REGION"] if region is None else region
    role_arn = os.environ["SAGEMAKER_ROLE"] if role is None else role

    offline_store_config_object = (
        {} if offline_store_config is None else offline_store_config
    )

    if s3_key is not None:
        offline_store_config_object["S3StorageConfig"] = {
            "S3Uri": f"s3://{data_lake_bucket_name}/{s3_key}"
        }

    sagemakerclient = boto3.client(
        "sagemaker",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    s3client = boto3.client(
        "s3",
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
            offline_store_config_object,
            role_arn,
            description,
        )

    if is_different:
        sagemakerclient.delete_feature_group(FeatureGroupName=feature_group_name)
        sleep(10)

    response = {"Message": "Feature group already exists with the provided definition."}

    try:
        s3client.create_bucket(Bucket=data_lake_bucket_name)
    except ClientError as e:
        pass

    if is_different or is_new:
        response = sagemakerclient.create_feature_group(
            FeatureGroupName=feature_group_name,
            RecordIdentifierFeatureName=record_identifier_feature,
            EventTimeFeatureName=event_time_feature,
            FeatureDefinitions=feature_definitions,
            OnlineStoreConfig=online_store_config,
            OfflineStoreConfig=offline_store_config_object,
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
    offline_store,
    role_arn,
    description,
):

    response = sagemakerclient.describe_feature_group(
        FeatureGroupName=feature_group_name
    )

    offline_store_config = offline_store

    if (
        "S3StorageConfig" in response["OfflineStoreConfig"]
        and "S3StorageConfig" in offline_store_config
    ):
        if (
            "ResolvedOutputS3Uri" not in offline_store_config
            or offline_store_config["S3StorageConfig"]["ResolvedOutputS3Uri"]
            == response["OfflineStoreConfig"]["S3StorageConfig"]["ResolvedOutputS3Uri"]
        ):
            offline_store_config["S3StorageConfig"]["ResolvedOutputS3Uri"] = response[
                "OfflineStoreConfig"
            ]["S3StorageConfig"]["ResolvedOutputS3Uri"]

    if (
        "DisableGlueTableCreation" not in offline_store_config
        or offline_store_config["DisableGlueTableCreation"]
        == response["OfflineStoreConfig"]["DisableGlueTableCreation"]
    ):
        offline_store_config["DisableGlueTableCreation"] = response[
            "OfflineStoreConfig"
        ]["DisableGlueTableCreation"]

    if (
        "DataCatalogConfig" not in offline_store_config
        or offline_store_config["DataCatalogConfig"]
        == response["OfflineStoreConfig"]["DataCatalogConfig"]
    ):
        offline_store_config["DataCatalogConfig"] = response["OfflineStoreConfig"][
            "DataCatalogConfig"
        ]

    return (
        response["RoleArn"] != role_arn
        or response["RecordIdentifierFeatureName"] != record_identifier_feature
        or response["EventTimeFeatureName"] != event_time_feature
        or response["FeatureDefinitions"] != feature_definitions
        or response["OnlineStoreConfig"] != online_store_config
        or response["OfflineStoreConfig"] != offline_store_config
        or response["Description"] != description
    )


def load_feature_group(
    name,
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    environment_prefix=None,
):
    """
    Loads a feature group object from the feature store by name
    """
    environment_prefix_name = environment_prefix

    aws_access_key_id = (
        os.environ["AWS_ACCESS_KEY"] if aws_access_key is None else aws_access_key
    )
    aws_secret_key_id = (
        os.environ["AWS_SECRET_KEY"] if aws_secret_key is None else aws_secret_key
    )

    environment_prefix_name = (
        os.environ["ENVIRONMENT_PREFIX"]
        if environment_prefix is None
        else environment_prefix
    )
    feature_group_name = name.replace("_", "-")
    region_name = boto3.Session().region_name if region is None else region
    boto_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_access_key_id,
        region_name=region_name,
    )

    sagemaker_client = boto_session.client(
        service_name="sagemaker", region_name=region_name
    )

    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime", region_name=region_name
    )

    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )

    return FeatureGroup(
        name=f"{feature_group_name}", sagemaker_session=feature_store_session
    )
