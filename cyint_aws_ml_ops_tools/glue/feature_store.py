import os
import re
from copy import deepcopy
from functools import lru_cache
from time import sleep

import boto3
from botocore.exceptions import ClientError
from botocore.session import get_session
from botocore.validate import validate_parameters
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

from cyint_aws_ml_ops_tools.universal.pipeline import create_s3bucket_if_not_exist, resolve_aws_credentials


def define_feature_group(
    name,
    description,
    record_identifier_feature,
    event_time_feature,
    feature_definitions,
    online_store_config=None,
    offline_store_config=None,
    s3_key=None,
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    role=None,
    data_lake_bucket=None,
    environment_prefix=None,
    *,
    allow_recreate=False,
):
    """
    Primary function to call when defining a new feature group in a Jupyter notebook or script. This function is designed to succeed regardless of
    existing state of the system.
    """
    if type(allow_recreate) is not bool:
        raise TypeError("allow_recreate must be an explicit boolean")

    environment_prefix_name = environment_prefix

    aws_access_key_id, aws_secret_key_id = resolve_aws_credentials(aws_access_key, aws_secret_key)

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

    region_name = region if region is not None else os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    role_arn = os.environ["SAGEMAKER_ROLE"] if role is None else role

    feature_group_name = f"{environment_prefix_name}{name}".replace("_", "-")
    request = prepare_feature_group_request(
        feature_group_name, description, record_identifier_feature, event_time_feature,
        feature_definitions, online_store_config, offline_store_config,
        s3_key, data_lake_bucket_name, role_arn,
    )
    sagemakerclient = boto3.client(
        "sagemaker", aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id, region_name=region_name,
    )
    is_new = is_feature_group_new(sagemakerclient, feature_group_name)
    is_different = False
    if not is_new:
        is_different = is_feature_group_different(
            sagemakerclient, feature_group_name, record_identifier_feature,
            event_time_feature, request["FeatureDefinitions"], request["OnlineStoreConfig"],
            request.get("OfflineStoreConfig", {}), role_arn, description,
        )
    if is_different and not allow_recreate:
        raise ValueError(
            "Feature group differs; explicit allow_recreate=True is required "
            "after reviewing data retention and recovery."
        )
    if not is_new and not is_different:
        return {"feature_group": {"Message": "Feature group already exists with the provided definition."}}

    s3client = boto3.client(
        "s3", aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id, region_name=region_name,
    )
    create_s3bucket_if_not_exist(data_lake_bucket_name, s3client)
    if is_different:
        sagemakerclient.delete_feature_group(FeatureGroupName=feature_group_name)
        wait_for_feature_group_deletion(sagemakerclient, feature_group_name)
    return {"feature_group": sagemakerclient.create_feature_group(**request)}




def wait_for_feature_group_deletion(sagemakerclient, feature_group_name, *, attempts=120, delay=10):
    """Poll up to 120 checks; only provider not-found proves deletion."""
    for attempt in range(attempts):
        try:
            response = sagemakerclient.describe_feature_group(FeatureGroupName=feature_group_name)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in {"ResourceNotFound", "ResourceNotFoundException"}:
                return
            raise
        if response.get("FeatureGroupStatus") == "DeleteFailed":
            raise RuntimeError("Feature group deletion failed; recreation stopped")
        if attempt + 1 < attempts:
            sleep(delay)
    raise TimeoutError("Feature group deletion was not confirmed; recreation stopped")


def is_feature_group_new(sagemakerclient, feature_group_name):
    try:
        response = sagemakerclient.describe_feature_group(
            FeatureGroupName=feature_group_name
        )
        return False
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"ResourceNotFound", "ResourceNotFoundException"}:
            return True
        raise


@lru_cache(maxsize=1)
def feature_group_create_shape():
    return get_session().get_service_model("sagemaker").operation_model("CreateFeatureGroup").input_shape


def validate_model_constraints(shape, value):
    """Check enums/patterns that botocore's structural validator does not check."""
    if shape.type_name == "structure":
        for key, item in value.items():
            validate_model_constraints(shape.members[key], item)
    elif shape.type_name == "list":
        for item in value:
            validate_model_constraints(shape.member, item)
    elif shape.type_name == "map":
        for key, item in value.items():
            validate_model_constraints(shape.key, key)
            validate_model_constraints(shape.value, item)
    elif shape.type_name == "string":
        if shape.enum and value not in shape.enum:
            raise ValueError("Feature-group request contains an unsupported enumerated value")
        pattern = shape.metadata.get("pattern")
        if pattern and re.fullmatch(pattern, value) is None:
            raise ValueError("Feature-group request does not match its service-model string pattern")


def prepare_feature_group_request(name, description, record_identifier, event_time, definitions,
                                  online_config, offline_config, s3_key, bucket, role_arn):
    online = {"EnableOnlineStore": True} if online_config is None else deepcopy(online_config)
    offline = {} if offline_config is None else deepcopy(offline_config)
    if not isinstance(online, dict) or not isinstance(offline, dict):
        raise TypeError("Feature-store configurations must be dictionaries")
    if s3_key is not None:
        storage = offline.setdefault("S3StorageConfig", {})
        if not isinstance(storage, dict):
            raise TypeError("S3StorageConfig must be a dictionary")
        storage["S3Uri"] = f"s3://{bucket}/{s3_key}"
    request = {
        "FeatureGroupName": name, "Description": description,
        "RecordIdentifierFeatureName": record_identifier, "EventTimeFeatureName": event_time,
        "FeatureDefinitions": deepcopy(definitions), "OnlineStoreConfig": online, "RoleArn": role_arn,
    }
    if offline:
        request["OfflineStoreConfig"] = offline
    shape = feature_group_create_shape()
    validate_parameters(request, shape)
    validate_model_constraints(shape, request)
    names = [definition["FeatureName"] for definition in request["FeatureDefinitions"]]
    if len(names) != len(set(names)) or any(name in {"is_deleted", "write_time", "api_invocation_time"} for name in names):
        raise ValueError("Feature names must be unique and must not use reserved names")
    if record_identifier not in names or event_time not in names:
        raise ValueError("Record and event-time identifiers must name defined features")
    time_definition = next(definition for definition in request["FeatureDefinitions"] if definition["FeatureName"] == event_time)
    if time_definition["FeatureType"] not in {"String", "Fractional"}:
        raise ValueError("Event-time feature must be String or Fractional")
    if not online.get("EnableOnlineStore", False) and not offline:
        raise ValueError("Enable an online or offline feature store")
    if offline.get("TableFormat") == "Iceberg" and offline.get("DisableGlueTableCreation", False):
        raise ValueError("Iceberg feature stores require Glue table creation")
    return request


def normalized_online(config):
    result = deepcopy(config or {})
    result.setdefault("EnableOnlineStore", False)
    if result["EnableOnlineStore"]:
        result.setdefault("StorageType", "Standard")
    else:
        result.pop("StorageType", None)
    return result


def normalized_offline(config):
    result = deepcopy(config or {})
    if not result:
        return result
    if "S3StorageConfig" in result:
        result["S3StorageConfig"].pop("ResolvedOutputS3Uri", None)
    result.setdefault("DisableGlueTableCreation", False)
    if result.get("TableFormat", "Default") == "Default":
        result["TableFormat"] = "Glue"
    return result



def describe_feature_group_complete(client, name):
    """Collect every feature-definition page before deciding a group differs."""
    response = client.describe_feature_group(FeatureGroupName=name)
    result = deepcopy(response)
    definitions = deepcopy(response["FeatureDefinitions"])
    token = response.get("NextToken")
    seen = set()
    stable_fields = ("FeatureGroupArn", "CreationTime", "FeatureGroupStatus", "RoleArn", "RecordIdentifierFeatureName",
                     "EventTimeFeatureName", "OnlineStoreConfig", "OfflineStoreConfig", "Description")
    while token:
        if not isinstance(token, str) or token in seen or len(seen) >= 1000:
            raise RuntimeError("Feature-definition pagination did not complete")
        seen.add(token)
        page = client.describe_feature_group(FeatureGroupName=name, NextToken=token)
        if any(page.get(field) != response.get(field) for field in stable_fields):
            raise RuntimeError("Feature-group configuration changed during enumeration")
        definitions.extend(deepcopy(page["FeatureDefinitions"]))
        token = page.get("NextToken")
    names = [definition["FeatureName"] for definition in definitions]
    if len(names) != len(set(names)):
        raise RuntimeError("Feature-definition enumeration returned duplicate names")
    result["FeatureDefinitions"] = definitions
    result["NextToken"] = ""
    return result


def is_feature_group_different(sagemakerclient, feature_group_name, record_identifier_feature,
                               event_time_feature, feature_definitions, online_store_config,
                               offline_store, role_arn, description):
    response = describe_feature_group_complete(sagemakerclient, feature_group_name)
    desired_offline = normalized_offline(offline_store)
    actual_offline = normalized_offline(response.get("OfflineStoreConfig", {}))
    # AWS generates these catalog values when the caller does not specify them.
    if "DataCatalogConfig" not in desired_offline:
        actual_offline.pop("DataCatalogConfig", None)
    return (
        response["RoleArn"] != role_arn
        or response["RecordIdentifierFeatureName"] != record_identifier_feature
        or response["EventTimeFeatureName"] != event_time_feature
        or response["FeatureDefinitions"] != feature_definitions
        or normalized_online(response.get("OnlineStoreConfig", {})) != normalized_online(online_store_config)
        or actual_offline != desired_offline
        or response.get("Description", "") != description
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

    aws_access_key_id, aws_secret_key_id = resolve_aws_credentials(aws_access_key, aws_secret_key)

    environment_prefix_name = (
        os.environ["ENVIRONMENT_PREFIX"]
        if environment_prefix is None
        else environment_prefix
    )
    feature_group_name = f"{environment_prefix_name}{name}".replace("_", "-")
    region_name = boto3.Session().region_name if region is None else region
    boto_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
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
