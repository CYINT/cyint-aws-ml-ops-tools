"""Real request/recreation sequence, caller isolation and no-write bad intent."""
from copy import deepcopy
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError, ParamValidationError
from botocore.stub import Stubber
import pytest

from cyint_aws_ml_ops_tools.glue import feature_store


ROLE = "arn:aws:iam::123456789012:role/qualification"
ARN = "arn:aws:sagemaker:us-east-1:123456789012:feature-group/qualification-features"
FEATURES = [{"FeatureName": "record", "FeatureType": "String"}, {"FeatureName": "time", "FeatureType": "Fractional"}]


def arguments(**overrides):
    return {**dict(name="features", description="qualification", record_identifier_feature="record",
                   event_time_feature="time", feature_definitions=deepcopy(FEATURES), aws_access_key="testing",
                   aws_secret_key="testing", region="us-east-1", role=ROLE, data_lake_bucket="data",
                   environment_prefix="qualification-"), **overrides}


def request(**overrides):
    return {**dict(FeatureGroupName="qualification-features", Description="qualification",
                   RecordIdentifierFeatureName="record", EventTimeFeatureName="time", FeatureDefinitions=deepcopy(FEATURES),
                   OnlineStoreConfig={"EnableOnlineStore": True}, RoleArn=ROLE), **overrides}


def existing(**overrides):
    return {**request(), "FeatureGroupArn": ARN, "CreationTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "FeatureGroupStatus": "Created", "NextToken": "", **overrides}


@pytest.mark.parametrize("overrides,exception", [
    ({"feature_definitions": "not-a-list"}, ParamValidationError),
    ({"feature_definitions": [{"FeatureName": "record"}]}, ParamValidationError),
    ({"online_store_config": {"EnableOnlineStore": "true"}}, ParamValidationError),
    ({"offline_store_config": {"S3StorageConfig": {"S3Uri": 42}}}, ParamValidationError),
    ({"offline_store_config": {"Unsupported": True}}, ParamValidationError),
    ({"description": None}, ParamValidationError),
    ({"role": "too-short"}, ParamValidationError),
    ({"role": "invalid-role-with-sufficient-length"}, ValueError),
    ({"feature_definitions": [{"FeatureName": "record", "FeatureType": "Unsupported"}]}, ValueError),
    ({"feature_definitions": FEATURES + [FEATURES[0]]}, ValueError),
    ({"feature_definitions": [{"FeatureName": "bad name", "FeatureType": "String"}]}, ValueError),
    ({"record_identifier_feature": "missing"}, ValueError),
    ({"feature_definitions": [FEATURES[0], {"FeatureName": "time", "FeatureType": "Integral"}]}, ValueError),
    ({"online_store_config": {}, "offline_store_config": {}}, ValueError),
    ({"offline_store_config": {"S3StorageConfig": {"S3Uri": "s3://qualification-data/features"}, "TableFormat": "Iceberg", "DisableGlueTableCreation": True}}, ValueError),
])
def test_malformed_opted_in_replacement_fails_before_client_or_write(mocker, overrides, exception):
    factory = mocker.patch.object(feature_store.boto3, "client")
    supplied = arguments(**deepcopy(overrides))
    before = deepcopy(supplied)
    with pytest.raises(exception):
        feature_store.define_feature_group(**supplied, allow_recreate=True)
    factory.assert_not_called()
    assert supplied == before


def test_real_sdk_online_only_new_group_request_omits_empty_offline_config(mocker):
    sm = boto3.client("sagemaker", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, s3])
    with Stubber(sm) as service, Stubber(s3) as storage:
        service.add_client_error("describe_feature_group", "ResourceNotFound", expected_params={"FeatureGroupName": "qualification-features"})
        storage.add_response("create_bucket", {}, {"Bucket": "qualification-data"})
        service.add_response("create_feature_group", {"FeatureGroupArn": ARN}, request())
        assert feature_store.define_feature_group(**arguments()) == {"feature_group": {"FeatureGroupArn": ARN}}
        service.assert_no_pending_responses()
        storage.assert_no_pending_responses()


def test_real_sdk_opted_in_recreate_uses_validated_request_after_confirmed_absence(mocker):
    sm = boto3.client("sagemaker", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, s3])
    pause = mocker.patch.object(feature_store, "sleep")
    desired = {"S3StorageConfig": {"S3Uri": "s3://qualification-data/features", "KmsKeyId": "alias/qualification"}}
    supplied = arguments(offline_store_config=desired)
    before = deepcopy(supplied)
    observed = deepcopy(desired)
    observed["S3StorageConfig"]["ResolvedOutputS3Uri"] = "s3://qualification-data/features/resolved"
    observed.update({"DataCatalogConfig": {"Catalog": "AwsDataCatalog", "Database": "qualification", "TableName": "qualification"}, "DisableGlueTableCreation": False})
    old = existing(Description="old", OfflineStoreConfig=observed)
    order = []
    sm.meta.events.register("before-parameter-build.*.*", lambda model, **kw: order.append(model.name))
    s3.meta.events.register("before-parameter-build.*.*", lambda model, **kw: order.append(model.name))
    with Stubber(sm) as service, Stubber(s3) as storage:
        service.add_response("describe_feature_group", old, {"FeatureGroupName": "qualification-features"})
        service.add_response("describe_feature_group", old, {"FeatureGroupName": "qualification-features"})
        storage.add_response("create_bucket", {}, {"Bucket": "qualification-data"})
        service.add_response("delete_feature_group", {}, {"FeatureGroupName": "qualification-features"})
        service.add_response("describe_feature_group", {**old, "FeatureGroupStatus": "Deleting"}, {"FeatureGroupName": "qualification-features"})
        service.add_client_error("describe_feature_group", "ResourceNotFound", expected_params={"FeatureGroupName": "qualification-features"})
        service.add_response("create_feature_group", {"FeatureGroupArn": ARN}, request(OfflineStoreConfig=desired))
        result = feature_store.define_feature_group(**supplied, allow_recreate=True)
        assert result["feature_group"]["FeatureGroupArn"] == ARN
        service.assert_no_pending_responses()
        storage.assert_no_pending_responses()
    assert order == ["DescribeFeatureGroup", "DescribeFeatureGroup", "CreateBucket", "DeleteFeatureGroup", "DescribeFeatureGroup", "DescribeFeatureGroup", "CreateFeatureGroup"]
    pause.assert_called_once_with(10)
    assert supplied == before
    assert "ResolvedOutputS3Uri" not in desired["S3StorageConfig"]


@pytest.mark.parametrize("different", [False, True])
def test_real_comparison_never_injects_provider_fields_or_mutates_caller(mocker, different):
    desired = {"S3StorageConfig": {"S3Uri": "s3://qualification-data/features"}}
    supplied = arguments(offline_store_config=desired)
    before = deepcopy(supplied)
    actual = deepcopy(desired)
    actual["S3StorageConfig"]["ResolvedOutputS3Uri"] = "s3://qualification-data/features/resolved"
    actual.update({"DataCatalogConfig": {"Catalog": "AwsDataCatalog", "Database": "qualification", "TableName": "qualification"}, "DisableGlueTableCreation": False, "TableFormat": "Glue"})
    sm = mocker.Mock()
    response = existing(Description="old" if different else "qualification", OfflineStoreConfig=actual,
                        OnlineStoreConfig={"EnableOnlineStore": True, "StorageType": "Standard"})
    original_response = deepcopy(response)
    sm.describe_feature_group.return_value = response
    factory = mocker.patch.object(feature_store.boto3, "client", return_value=sm)
    if different:
        with pytest.raises(ValueError, match="explicit allow_recreate"):
            feature_store.define_feature_group(**supplied)
    else:
        assert "already exists" in feature_store.define_feature_group(**supplied)["feature_group"]["Message"]
    assert supplied == before
    assert response == original_response
    factory.assert_called_once()
    sm.delete_feature_group.assert_not_called()
    sm.create_feature_group.assert_not_called()


def test_default_offline_store_destination_cannot_leak_across_calls(mocker):
    sm1, sm2, storage1, storage2 = (mocker.Mock() for _ in range(4))
    for sm in (sm1, sm2):
        sm.describe_feature_group.side_effect = ClientError({"Error": {"Code": "ResourceNotFound"}}, "DescribeFeatureGroup")
    for storage in (storage1, storage2):
        storage.meta.region_name = "us-east-1"
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm1, storage1, sm2, storage2])
    feature_store.define_feature_group(**arguments(name="first", data_lake_bucket="first", s3_key="prefix"))
    feature_store.define_feature_group(**arguments(name="second", data_lake_bucket="second"))
    assert sm1.create_feature_group.call_args.kwargs["OfflineStoreConfig"]["S3StorageConfig"]["S3Uri"] == "s3://qualification-first/prefix"
    assert "OfflineStoreConfig" not in sm2.create_feature_group.call_args.kwargs


def test_derived_s3_prefix_preserves_explicit_encryption_configuration(mocker):
    supplied_offline = {"S3StorageConfig": {"S3Uri": "s3://qualification-old/prefix", "KmsKeyId": "alias/qualification"}}
    before = deepcopy(supplied_offline)
    sm, storage = mocker.Mock(), mocker.Mock()
    sm.describe_feature_group.side_effect = ClientError({"Error": {"Code": "ResourceNotFound"}}, "DescribeFeatureGroup")
    storage.meta.region_name = "us-east-1"
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, storage])
    feature_store.define_feature_group(**arguments(offline_store_config=supplied_offline, s3_key="new-prefix"))
    assert sm.create_feature_group.call_args.kwargs["OfflineStoreConfig"]["S3StorageConfig"] == {
        "S3Uri": "s3://qualification-data/new-prefix", "KmsKeyId": "alias/qualification",
    }
    assert supplied_offline == before


def test_comparison_reads_all_feature_definition_pages_before_no_change_decision(mocker):
    sm = boto3.client("sagemaker", region_name="us-east-1")
    factory = mocker.patch.object(feature_store.boto3, "client", return_value=sm)
    first = existing(FeatureDefinitions=FEATURES[:1], NextToken="second-page")
    second = existing(FeatureDefinitions=FEATURES[1:])
    with Stubber(sm) as service:
        service.add_response("describe_feature_group", first, {"FeatureGroupName": "qualification-features"})
        service.add_response("describe_feature_group", first, {"FeatureGroupName": "qualification-features"})
        service.add_response("describe_feature_group", second, {"FeatureGroupName": "qualification-features", "NextToken": "second-page"})
        result = feature_store.define_feature_group(**arguments())
        assert "already exists" in result["feature_group"]["Message"]
        service.assert_no_pending_responses()
    factory.assert_called_once()  # No S3 client or write for an unchanged group.


@pytest.mark.parametrize("failure", ["repeated-token", "metadata-change", "duplicate-feature", "denied-page"])
def test_incomplete_or_unstable_feature_pagination_stops_before_write(mocker, failure):
    sm = mocker.Mock()
    first = existing(FeatureDefinitions=FEATURES[:1], NextToken="second-page")
    second = existing(FeatureDefinitions=FEATURES[1:])
    if failure == "repeated-token":
        second["NextToken"] = "second-page"
    elif failure == "metadata-change":
        second["Description"] = "changed-during-enumeration"
    elif failure == "duplicate-feature":
        second["FeatureDefinitions"] = FEATURES[:1]
    else:
        second = ClientError({"Error": {"Code": "AccessDeniedException"}}, "DescribeFeatureGroup")
    sm.describe_feature_group.side_effect = [first, first, second]
    factory = mocker.patch.object(feature_store.boto3, "client", return_value=sm)
    with pytest.raises((RuntimeError, ClientError)):
        feature_store.define_feature_group(**arguments(), allow_recreate=True)
    factory.assert_called_once()
    sm.delete_feature_group.assert_not_called()
    sm.create_feature_group.assert_not_called()


def test_online_only_group_without_unused_role_metadata_is_idempotent(mocker):
    sm = boto3.client("sagemaker", region_name="us-east-1")
    factory = mocker.patch.object(feature_store.boto3, "client", return_value=sm)
    response = existing()
    response.pop("RoleArn")  # Role is optional and used for the offline store.
    with Stubber(sm) as service:
        service.add_response("describe_feature_group", response, {"FeatureGroupName": "qualification-features"})
        service.add_response("describe_feature_group", response, {"FeatureGroupName": "qualification-features"})
        result = feature_store.define_feature_group(**arguments())
        assert "already exists" in result["feature_group"]["Message"]
        service.assert_no_pending_responses()
    factory.assert_called_once()


def test_missing_offline_role_is_unknown_not_permission_to_recreate(mocker):
    sm = mocker.Mock()
    offline = {"S3StorageConfig": {"S3Uri": "s3://qualification-data/features"}}
    response = existing(OfflineStoreConfig=deepcopy(offline))
    response.pop("RoleArn")
    sm.describe_feature_group.return_value = response
    factory = mocker.patch.object(feature_store.boto3, "client", return_value=sm)
    with pytest.raises(RuntimeError, match="execution role is unconfirmed"):
        feature_store.define_feature_group(**arguments(offline_store_config=offline), allow_recreate=True)
    factory.assert_called_once()
    sm.delete_feature_group.assert_not_called()
    sm.create_feature_group.assert_not_called()
