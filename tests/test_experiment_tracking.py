"""Service-model validated requests; Stubber consumes every call before transport."""
from datetime import datetime, timezone
from types import SimpleNamespace

import boto3
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber
import pytest

from cyint_aws_ml_ops_tools.experimentation import _tracking, experiment
from cyint_aws_ml_ops_tools.universal.pipeline import create_s3bucket_if_not_exist


@pytest.fixture
def client():
    return boto3.client("sagemaker", region_name="us-east-1")


@pytest.fixture
def component(mocker):
    mocker.patch.object(_tracking, "uuid4", return_value=SimpleNamespace(hex="a" * 32))
    return "cyint-metadata-" + "a" * 32


def tracker(client):
    return _tracking.ParameterTracker(
        display_name="metadata", artifact_bucket="qualification-bucket",
        artifact_prefix="model/run", sagemaker_boto_client=client,
    )


def start(stub, component, display="metadata"):
    stub.add_response("create_trial_component", {}, {
        "TrialComponentName": component, "DisplayName": display,
        "Status": {"PrimaryStatus": "InProgress"}, "StartTime": ANY,
    })


def end(stub, component, parameters, failed=False):
    status = {"PrimaryStatus": "Failed", "Message": "Tracking context failed"} if failed else {"PrimaryStatus": "Completed"}
    stub.add_response("update_trial_component", {}, {
        "TrialComponentName": component, "Status": status,
        "EndTime": ANY, "Parameters": parameters,
    })


def test_experiment_trial_and_component_association_use_current_service_model(client):
    with Stubber(client) as stub:
        stub.add_response("create_experiment", {}, {"ExperimentName": "qualification", "Description": "hypothesis"})
        stub.add_response("create_trial", {}, {"TrialName": "qualification-trial", "ExperimentName": "qualification"})
        stub.add_response("associate_trial_component", {}, {"TrialName": "qualification-trial", "TrialComponentName": "metadata"})
        exp = _tracking.create_experiment(experiment_name="qualification", description="hypothesis", sagemaker_boto_client=client)
        trial = _tracking.create_trial(trial_name="qualification-trial", experiment_name=exp.experiment_name, sagemaker_boto_client=client)
        assert exp.experiment_name == "qualification"
        assert trial.trial_name == "qualification-trial"
        trial.add_trial_component(_tracking.TrialComponentReference("metadata"))
        stub.assert_no_pending_responses()


def test_tracker_preserves_scalars_and_serializes_grid_metadata(client, component):
    with Stubber(client) as stub:
        start(stub, component)
        end(stub, component, {
            "name": {"StringValue": "model"}, "epochs": {"NumberValue": 2.0},
            "rates": {"StringValue": "[0.1, 0.2]"}, "enabled": {"StringValue": "true"},
        })
        with tracker(client) as context:
            context.log_parameters({"name": "model", "epochs": 1})
            context.log_parameters({"epochs": 2, "rates": [0.1, 0.2], "enabled": True})
        assert context.trial_component.trial_component_name == component
        assert context.artifact_bucket == "qualification-bucket"
        stub.assert_no_pending_responses()


def test_context_failure_is_propagated_without_copying_exception_text_to_cloud(client, component):
    failure = ValueError("sensitive-detail-not-for-cloud")
    with Stubber(client) as stub:
        start(stub, component)
        end(stub, component, {}, failed=True)
        with pytest.raises(ValueError) as caught:
            with tracker(client):
                raise failure
        assert caught.value is failure
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_invalid_parameters_do_not_partially_update_tracker(client, component, value):
    with Stubber(client) as stub:
        start(stub, component)
        end(stub, component, {})
        with tracker(client) as context:
            with pytest.raises((TypeError, ValueError)):
                context.log_parameters({"valid": 1, "invalid": value})
        stub.assert_no_pending_responses()


def test_tracker_context_and_logging_cannot_be_replayed(client, component):
    context = tracker(client)
    with pytest.raises(RuntimeError):
        context.log_parameters({"value": 1})
    with Stubber(client) as stub:
        start(stub, component)
        end(stub, component, {})
        with context:
            pass
        with pytest.raises(RuntimeError):
            context.__enter__()
        with pytest.raises(RuntimeError):
            context.log_parameters({"value": 1})
        with pytest.raises(RuntimeError):
            context.__exit__(None, None, None)
        stub.assert_no_pending_responses()


def test_provider_error_is_not_treated_as_created_experiment(client):
    with Stubber(client) as stub:
        stub.add_client_error("create_experiment", "AccessDeniedException", expected_params={"ExperimentName": "qualification", "Description": "hypothesis"})
        with pytest.raises(ClientError) as caught:
            _tracking.create_experiment(experiment_name="qualification", description="hypothesis", sagemaker_boto_client=client)
        assert caught.value.response["Error"]["Code"] == "AccessDeniedException"
        stub.assert_no_pending_responses()


def test_trial_creation_preserves_training_callback_contract(client, component, mocker):
    mocker.patch.object(experiment.time, "time", return_value=1700000000)
    pause = mocker.patch.object(experiment.time, "sleep")
    callback = mocker.Mock()
    trial = "trial-dataset-0-hparams-0-1700000000"
    with Stubber(client) as stub:
        start(stub, component, display="trial-metadata-1700000000")
        end(stub, component, {"epochs": {"NumberValue": 2.0}, "rate": {"NumberValue": 0.1}})
        stub.add_response("create_trial", {}, {"TrialName": trial, "ExperimentName": "qualification"})
        stub.add_response("associate_trial_component", {}, {"TrialName": trial, "TrialComponentName": "experiment-metadata"})
        stub.add_response("associate_trial_component", {}, {"TrialName": trial, "TrialComponentName": component})
        experiment.create_trials(
            [{"row": 1}], [{"rate": 0.1}], {"epochs": 2},
            _tracking.ExperimentReference("qualification"), "qualification-bucket", "model", "1234567890", [],
            SimpleNamespace(trial_component=_tracking.TrialComponentReference("experiment-metadata")),
            callback, "qualification", client, "qualification-role", 0, 2, "us-east-1",
        )
        job = "qualification-model-12345678-dataset-0-hparams-0-1700000000"
        callback.assert_called_once_with(
            {"row": 1}, f"s3://qualification-bucket/model/1234567890/{trial}",
            "s3://qualification-bucket/code", "qualification-role", [], client,
            {"epochs": 2, "rate": 0.1}, job,
            {"ExperimentName": "qualification", "TrialName": trial, "TrialComponentDisplayName": job},
            "qualification-bucket", 2, "us-east-1",
        )
        assert pause.call_count == 2
        stub.assert_no_pending_responses()


def test_experiment_metadata_tracker_preserves_grid_and_reference(client, component):
    with Stubber(client) as stub:
        start(stub, component, display="qualification")
        end(stub, component, {"epochs": {"NumberValue": 2.0}, "rate": {"StringValue": "[0.1, 0.2]"}})
        reference = experiment.track_experiment_metadata(
            _tracking.ExperimentReference("qualification"), "qualification-bucket", "model", "run",
            {"epochs": 2}, {"rate": [0.1, 0.2]}, client,
        )
        assert reference.trial_component.trial_component_name == component
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("region", ["us-east-1", "us-west-2"])
def test_bucket_creation_uses_correct_regional_request(region):
    s3 = boto3.client("s3", region_name=region)
    request = {"Bucket": "qualification-bucket"}
    if region != "us-east-1":
        request["CreateBucketConfiguration"] = {"LocationConstraint": region}
    with Stubber(s3) as stub:
        stub.add_response("create_bucket", {}, request)
        create_s3bucket_if_not_exist("qualification-bucket", s3)
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("code,accepted", [("BucketAlreadyOwnedByYou", True), ("BucketAlreadyExists", False), ("AccessDenied", False)])
def test_bucket_errors_do_not_hide_other_owner_or_denied_access(code, accepted):
    s3 = boto3.client("s3", region_name="us-east-1")
    with Stubber(s3) as stub:
        stub.add_client_error("create_bucket", code, expected_params={"Bucket": "qualification-bucket"})
        if accepted:
            create_s3bucket_if_not_exist("qualification-bucket", s3)
        else:
            with pytest.raises(ClientError):
                create_s3bucket_if_not_exist("qualification-bucket", s3)
        stub.assert_no_pending_responses()


def test_define_experiment_preserves_whole_training_path_without_live_calls(client, mocker):
    s3 = boto3.client("s3", region_name="us-east-1")
    mocker.patch.object(experiment.boto3, "Session", return_value=SimpleNamespace(
        client=lambda service: {"sagemaker": client, "s3": s3}[service],
    ))
    mocker.patch.object(_tracking, "uuid4", side_effect=[SimpleNamespace(hex="a" * 32), SimpleNamespace(hex="b" * 32)])
    mocker.patch.object(experiment.time, "time", return_value=1700000000)
    mocker.patch.object(experiment.time, "sleep")
    callback = mocker.Mock()
    component1 = "cyint-metadata-" + "a" * 32
    component2 = "cyint-metadata-" + "b" * 32
    trial = "trial-dataset-0-hparams-0-1700000000"
    with Stubber(s3) as storage, Stubber(client) as sm:
        storage.add_response("create_bucket", {}, {"Bucket": "qualification-bucket"})
        sm.add_response("create_experiment", {}, {"ExperimentName": "qualification-model-run", "Description": "hypothesis"})
        start(sm, component1, display="qualification-model-run")
        end(sm, component1, {"epochs": {"NumberValue": 2.0}, "rate": {"StringValue": "[0.1]"}})
        start(sm, component2, display="trial-metadata-1700000000")
        end(sm, component2, {"epochs": {"NumberValue": 2.0}, "rate": {"NumberValue": 0.1}})
        sm.add_response("create_trial", {}, {"TrialName": trial, "ExperimentName": "qualification-model-run"})
        sm.add_response("associate_trial_component", {}, {"TrialName": trial, "TrialComponentName": component1})
        sm.add_response("associate_trial_component", {}, {"TrialName": trial, "TrialComponentName": component2})
        run = experiment.define_experiment(
            "model", "hypothesis", callback, metrics=[], static_hyperparams={"epochs": 2},
            hyperparam_configuration={"rate": [0.1]}, dataset=[{"row": 1}], run="run",
            aws_access_key="testing", aws_secret_key="testing", region="us-east-1",
            role="qualification-role", environment_prefix="qualification", experiment_bucket="bucket", trial_wait_time=0,
        )
        assert run == "run"
        assert callback.call_count == 1
        assert callback.call_args.args[8] == {
            "ExperimentName": "qualification-model-run", "TrialName": trial,
            "TrialComponentDisplayName": "qualification-model-run-dataset-0-hparams-0-1700000000",
        }
        storage.assert_no_pending_responses()
        sm.assert_no_pending_responses()
