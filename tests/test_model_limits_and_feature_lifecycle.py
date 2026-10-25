"""Model maxima and supported ready/collection intent must gate all writes."""
from copy import deepcopy

import boto3
from botocore.exceptions import ParamValidationError
from botocore.stub import Stubber
import pytest

from cyint_aws_ml_ops_tools.glue import feature_store, jobs
from test_feature_group_public_preflight import arguments, existing, request, FEATURES, ARN, ROLE


@pytest.mark.parametrize("status", ["Creating", "Deleting", "CreateFailed", "DeleteFailed", "unknown", None])
@pytest.mark.parametrize("opt_in", [False, True])
def test_nonready_existing_group_is_neither_ready_nor_permission_to_mutate(mocker, status, opt_in):
    sm = mocker.Mock()
    sm.describe_feature_group.return_value = existing(FeatureGroupStatus=status)
    factory = mocker.patch.object(feature_store.boto3, "client", return_value=sm)
    with pytest.raises(RuntimeError, match="confirmed Created"):
        feature_store.define_feature_group(**arguments(), allow_recreate=opt_in)
    factory.assert_called_once()
    sm.delete_feature_group.assert_not_called()
    sm.create_feature_group.assert_not_called()


def test_nonready_differing_group_cannot_be_deleted_even_with_opt_in(mocker):
    sm = mocker.Mock()
    sm.describe_feature_group.return_value = existing(FeatureGroupStatus="Creating", Description="old")
    factory = mocker.patch.object(feature_store.boto3, "client", return_value=sm)
    with pytest.raises(RuntimeError, match="confirmed Created"):
        feature_store.define_feature_group(**arguments(), allow_recreate=True)
    factory.assert_called_once()
    sm.delete_feature_group.assert_not_called()


@pytest.mark.parametrize("change", [
    {"description": "x" * 129},
    {"name": "x" * 65},
    {"feature_definitions": FEATURES + [{"FeatureName": "x" * 65, "FeatureType": "String"}]},
    {"feature_definitions": FEATURES + [{"FeatureName": "f" + str(i), "FeatureType": "String"} for i in range(2499)]},
])
def test_oversized_feature_intent_stops_before_client_or_delete(mocker, change):
    factory = mocker.patch.object(feature_store.boto3, "client")
    with pytest.raises(ValueError, match="maximum"):
        feature_store.define_feature_group(**arguments(**deepcopy(change)), allow_recreate=True)
    factory.assert_not_called()


def test_exact_feature_description_name_and_list_maxima_are_accepted():
    definitions = FEATURES + [{"FeatureName": "f" + str(i), "FeatureType": "String"} for i in range(2498)]
    prepared = feature_store.prepare_feature_group_request(
        "x" * 64, "x" * 128, "record", "time", definitions, None, None, None, "qualification-data", ROLE,
    )
    assert len(prepared["FeatureDefinitions"]) == 2500
    assert len(prepared["Description"]) == 128


@pytest.mark.parametrize("kind", ["job", "trigger"])
def test_oversized_glue_description_stops_before_lookup_upload_or_job_mutation(mocker, kind):
    glue, s3 = mocker.Mock(), mocker.Mock()
    mocker.patch.object(jobs.boto3, "client", side_effect=[glue, s3])
    kwargs = {"job_configuration" if kind == "job" else "trigger_definition": {"Description": "x" * 2049}}
    with pytest.raises(ValueError, match="maximum"):
        jobs.define_job("job", "glueetl", "script.py", "script.py", "trigger", "ON_DEMAND",
                        aws_access_key="testing", aws_secret_key="testing", region="us-east-1",
                        role="qualification", glue_script_bucket="scripts", environment_prefix="qualification-", **kwargs)
    assert glue.mock_calls == [] and s3.mock_calls == []


def test_exact_glue_description_maxima_are_accepted():
    create, update = jobs.prepare_job_requests("job", "glueetl", "s3://qualification/script.py", "qualification", "3", {"Description": "x" * 2048})
    assert len(update["JobUpdate"]["Description"]) == 2048
    create, update = jobs.prepare_trigger_requests("job", "trigger", "ON_DEMAND", {"Description": "x" * 2048})
    assert len(create["Description"]) == 2048


def test_glue_utf16_model_pattern_preserves_valid_supplementary_unicode():
    create, update = jobs.prepare_job_requests("qualification-🚀", "glueetl", "s3://qualification/script.py", "qualification", "3", {"Description": "description 🚀\nnext line"})
    assert create["Name"] == "qualification-🚀"
    assert update["JobUpdate"]["Description"].endswith("next line")


@pytest.mark.parametrize("name", ["invalid\x00name", "invalid\nname"])
def test_glue_pattern_translation_does_not_admit_invalid_name_controls(name):
    with pytest.raises(ValueError, match="string pattern"):
        jobs.prepare_job_requests(name, "glueetl", "s3://qualification/script.py", "qualification", "3", {})


def vector(**overrides):
    return {"FeatureName": "embedding", "FeatureType": "Fractional", "CollectionType": "Vector",
            "CollectionConfig": {"VectorConfig": {"Dimension": 3}}, **overrides}


@pytest.mark.parametrize("feature", [
    vector(FeatureType="String"), vector(FeatureType="Integral"),
    {"FeatureName": "embedding", "FeatureType": "Fractional", "CollectionType": "Vector"},
    vector(CollectionConfig={"VectorConfig": {"Dimension": 8193}}),
    vector(CollectionConfig={"VectorConfig": {"Dimension": 0}}),
    vector(CollectionConfig={"VectorConfig": {"Dimension": True}}),
    vector(CollectionType="List"), vector(CollectionType="Set"),
])
def test_invalid_collection_intent_stops_before_client_or_delete(mocker, feature):
    factory = mocker.patch.object(feature_store.boto3, "client")
    with pytest.raises((ValueError, ParamValidationError)):
        feature_store.define_feature_group(**arguments(
            feature_definitions=FEATURES + [deepcopy(feature)],
            online_store_config={"EnableOnlineStore": True, "StorageType": "InMemory"},
        ), allow_recreate=True)
    factory.assert_not_called()


@pytest.mark.parametrize("online", [
    {"EnableOnlineStore": True}, {"EnableOnlineStore": True, "StorageType": "Standard"},
    {"EnableOnlineStore": True, "StorageType": "Standard_V2"},
    {"EnableOnlineStore": False, "StorageType": "InMemory"},
])
def test_collection_requires_enabled_inmemory_tier_before_any_write(mocker, online):
    factory = mocker.patch.object(feature_store.boto3, "client")
    with pytest.raises(ValueError, match="InMemory"):
        feature_store.define_feature_group(**arguments(
            feature_definitions=FEATURES + [vector()], online_store_config=online,
            offline_store_config={"S3StorageConfig": {"S3Uri": "s3://qualification-data/features"}},
        ), allow_recreate=True)
    factory.assert_not_called()


def test_multiple_vectors_are_not_silently_accepted(mocker):
    factory = mocker.patch.object(feature_store.boto3, "client")
    with pytest.raises(ValueError, match="one Vector"):
        feature_store.define_feature_group(**arguments(
            feature_definitions=FEATURES + [vector(), vector(FeatureName="second")],
            online_store_config={"EnableOnlineStore": True, "StorageType": "InMemory"},
        ), allow_recreate=True)
    factory.assert_not_called()


@pytest.mark.parametrize("dimension", [1, 8192])
def test_valid_fractional_vector_dimension_boundaries_forward_real_sdk_request(mocker, dimension):
    sm, s3 = boto3.client("sagemaker", region_name="us-east-1"), boto3.client("s3", region_name="us-east-1")
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, s3])
    definitions = FEATURES + [vector(CollectionConfig={"VectorConfig": {"Dimension": dimension}})]
    online = {"EnableOnlineStore": True, "StorageType": "InMemory"}
    with Stubber(sm) as service, Stubber(s3) as storage:
        service.add_client_error("describe_feature_group", "ResourceNotFound", expected_params={"FeatureGroupName": "qualification-features"})
        storage.add_response("create_bucket", {}, {"Bucket": "qualification-data"})
        service.add_response("create_feature_group", {"FeatureGroupArn": ARN}, request(FeatureDefinitions=definitions, OnlineStoreConfig=online))
        feature_store.define_feature_group(**arguments(feature_definitions=definitions, online_store_config=online))
        service.assert_no_pending_responses()
        storage.assert_no_pending_responses()


@pytest.mark.parametrize("collection", ["List", "Set"])
def test_supported_nonvector_collections_remain_available(collection):
    definitions = FEATURES + [{"FeatureName": "labels", "FeatureType": "String", "CollectionType": collection}]
    prepared = feature_store.prepare_feature_group_request("qualification-features", "qualification", "record", "time",
        definitions, {"EnableOnlineStore": True, "StorageType": "InMemory"}, None, None, "qualification-data", ROLE)
    assert prepared["FeatureDefinitions"][-1]["CollectionType"] == collection


def test_iceberg_fractional_time_is_rejected_before_any_client(mocker):
    factory = mocker.patch.object(feature_store.boto3, "client")
    with pytest.raises(ValueError, match="Iceberg event-time"):
        feature_store.define_feature_group(**arguments(offline_store_config={
            "S3StorageConfig": {"S3Uri": "s3://qualification-data/features"}, "TableFormat": "Iceberg",
        }), allow_recreate=True)
    factory.assert_not_called()


def test_valid_iceberg_string_time_remains_supported():
    definitions = [FEATURES[0], {"FeatureName": "time", "FeatureType": "String"}]
    prepared = feature_store.prepare_feature_group_request("qualification-features", "qualification", "record", "time",
        definitions, None, {"S3StorageConfig": {"S3Uri": "s3://qualification-data/features"}, "TableFormat": "Iceberg"},
        None, "qualification-data", ROLE)
    assert prepared["OfflineStoreConfig"]["TableFormat"] == "Iceberg"
