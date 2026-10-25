"""Real SDK imports and offline public API safety contracts."""
import importlib
import socket
from types import SimpleNamespace

import pytest
from sagemaker.feature_store.feature_group import FeatureGroup

from cyint_aws_ml_ops_tools.glue import feature_store
from cyint_aws_ml_ops_tools.universal.pipeline import prepare_pipeline_variables, sanitize_bucket_name


@pytest.mark.parametrize("module", [
    "deployment.inference", "experimentation.experiment", "experimentation.features",
    "glue.feature_store", "glue.jobs", "universal.pipeline",
])
def test_all_library_modules_import_with_current_real_sdks(module):
    assert importlib.import_module("cyint_aws_ml_ops_tools." + module) is not None


def test_feature_group_constructs_with_real_current_sdk_without_network():
    group = feature_store.load_feature_group(
        "features", aws_access_key="testing", aws_secret_key="testing",
        region="us-east-1", environment_prefix="qualification-",
    )
    assert isinstance(group, FeatureGroup)
    assert group.name == "qualification-features"
    assert group.sagemaker_session.sagemaker_client.meta.service_model.service_name == "sagemaker"


def test_explicit_pipeline_credentials_override_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY", "different-dummy-key")
    monkeypatch.setenv("AWS_SECRET_KEY", "different-dummy-secret")
    assert prepare_pipeline_variables("prefix", "key", "secret", "us-west-2") == [
        "prefix", "key", "secret", "us-west-2",
    ]


def test_pipeline_variables_support_existing_environment_contract():
    assert prepare_pipeline_variables() == ["qualification", "testing", "testing", "us-east-1"]


def test_bucket_name_normalization():
    assert sanitize_bucket_name("qualification_bucket") == "qualification-bucket"


def _definition():
    return dict(
        name="features", description="qualification", record_identifier_feature="record",
        event_time_feature="time", feature_definitions=[], online_store_config={"EnableOnlineStore": True},
        offline_store_config={}, aws_access_key="testing", aws_secret_key="testing", region="us-east-1",
        role="arn:aws:iam::123456789012:role/qualification", data_lake_bucket="data",
        environment_prefix="qualification-",
    )


def _changed_group(mocker):
    sm = mocker.Mock()
    s3 = mocker.Mock()
    s3.meta = SimpleNamespace(region_name="us-east-1")
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, s3])
    mocker.patch.object(feature_store, "is_feature_group_new", return_value=False)
    mocker.patch.object(feature_store, "is_feature_group_different", return_value=True)
    return sm, s3


def test_changed_group_refuses_deletion_without_explicit_opt_in(mocker):
    sm, s3 = _changed_group(mocker)
    with pytest.raises(ValueError, match="explicit allow_recreate=True"):
        feature_store.define_feature_group(**_definition())
    sm.delete_feature_group.assert_not_called()
    sm.create_feature_group.assert_not_called()
    s3.create_bucket.assert_not_called()


def test_explicit_opt_in_preserves_recreate_contract(mocker):
    sm, s3 = _changed_group(mocker)
    wait = mocker.patch.object(feature_store, "wait_for_feature_group_deletion")
    result = feature_store.define_feature_group(**_definition(), allow_recreate=True)
    sm.delete_feature_group.assert_called_once_with(FeatureGroupName="qualification-features")
    sm.create_feature_group.assert_called_once()
    wait.assert_called_once_with(sm, "qualification-features")
    assert result == {"feature_group": sm.create_feature_group.return_value}


@pytest.mark.parametrize("value", ["true", 1, None])
def test_recreate_opt_in_rejects_ambiguous_values_before_client_access(mocker, value):
    factory = mocker.patch.object(feature_store.boto3, "client")
    with pytest.raises(TypeError, match="explicit boolean"):
        feature_store.define_feature_group(**_definition(), allow_recreate=value)
    factory.assert_not_called()


def test_qualification_network_boundary_blocks_before_connect():
    with pytest.raises(PermissionError, match="Offline qualification"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)


@pytest.mark.parametrize("function,method,not_found", [
    ("is_feature_group_new", "describe_feature_group", "ResourceNotFound"),
    ("is_new_job", "get_job", "EntityNotFoundException"),
    ("is_new_trigger", "get_trigger", "EntityNotFoundException"),
])
@pytest.mark.parametrize("code", ["AccessDeniedException", "ThrottlingException", "ValidationException"])
def test_unknown_or_denied_resource_state_never_means_missing(mocker, function, method, not_found, code):
    from botocore.exceptions import ClientError
    from cyint_aws_ml_ops_tools.glue import jobs
    client = mocker.Mock()
    getattr(client, method).side_effect = ClientError({"Error": {"Code": code}}, method)
    target = getattr(feature_store if function == "is_feature_group_new" else jobs, function)
    with pytest.raises(ClientError):
        target(client, "qualification")


@pytest.mark.parametrize("function,method,code", [
    ("is_feature_group_new", "describe_feature_group", "ResourceNotFound"),
    ("is_feature_group_new", "describe_feature_group", "ResourceNotFoundException"),
    ("is_new_job", "get_job", "EntityNotFoundException"),
    ("is_new_trigger", "get_trigger", "EntityNotFoundException"),
])
def test_only_provider_not_found_codes_allow_new_resource_state(mocker, function, method, code):
    from botocore.exceptions import ClientError
    from cyint_aws_ml_ops_tools.glue import jobs
    client = mocker.Mock()
    getattr(client, method).side_effect = ClientError({"Error": {"Code": code}}, method)
    target = getattr(feature_store if function == "is_feature_group_new" else jobs, function)
    assert target(client, "qualification") is True



def test_inference_entry_point_constructs_real_sdk_model_and_preserves_deploy_arguments(mocker):
    from cyint_aws_ml_ops_tools.deployment import inference
    deploy = mocker.patch.object(inference.Model, "deploy", autospec=True, return_value="qualification-result")
    result = inference.define_inference_endpoint(
        "model", "example.invalid/model:qualification", initial_instance_count=1,
        instance_type="ml.m5.large", aws_access_key="testing", aws_secret_key="testing",
        region="us-east-1", role="arn:aws:iam::123456789012:role/qualification", environment_prefix="qualification",
    )
    assert result == "qualification-result"
    model = deploy.call_args.args[0]
    assert model.model_data == "s3://qualification-model/model.tar.gz"
    assert deploy.call_args.kwargs["endpoint_name"] == "qualification-model-endpoint"
    assert deploy.call_args.kwargs["instance_type"] == "ml.m5.large"
    assert deploy.call_args.kwargs["initial_instance_count"] == 1
