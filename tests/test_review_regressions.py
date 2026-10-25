"""Offline regressions for independently reported integrity and S3 safety defects."""
import os
import subprocess
import sys
from types import SimpleNamespace

import boto3
from botocore.exceptions import ClientError
from botocore.stub import Stubber
import pytest

from cyint_aws_ml_ops_tools.glue import jobs, feature_store
from cyint_aws_ml_ops_tools.experimentation import experiment, _tracking
from cyint_aws_ml_ops_tools.universal.pipeline import resolve_aws_credentials, prepare_pipeline_variables


def error(code, operation="CreateBucket"):
    return ClientError({"Error": {"Code": code}}, operation)


@pytest.mark.parametrize("code", ["AccessDenied", "BucketAlreadyExists", "InvalidLocationConstraint"])
def test_script_bucket_failure_prevents_upload_and_job_mutation(mocker, code):
    s3 = boto3.client("s3", region_name="us-west-2")
    glue = mocker.Mock()
    upload = mocker.patch.object(s3, "upload_file")
    mocker.patch.object(jobs.boto3, "client", side_effect=[glue, s3])
    with Stubber(s3) as stub:
        stub.add_client_error("create_bucket", code, expected_params={
            "Bucket": "qualification-scripts",
            "CreateBucketConfiguration": {"LocationConstraint": "us-west-2"},
        })
        with pytest.raises(ClientError):
            jobs.define_job("job", "glueetl", "script.py", "script.py", "trigger", "ON_DEMAND",
                            aws_access_key="testing", aws_secret_key="testing", region="us-west-2",
                            role="qualification", glue_script_bucket="scripts", environment_prefix="qualification-")
        stub.assert_no_pending_responses()
    upload.assert_not_called()
    assert glue.mock_calls == []


@pytest.mark.parametrize("region,existing", [("us-east-1", False), ("us-west-2", False), ("us-west-2", True)])
def test_script_upload_uses_regional_request_and_preserves_destination(mocker, region, existing):
    s3 = boto3.client("s3", region_name=region)
    upload = mocker.patch.object(s3, "upload_file", return_value=None)
    request = {"Bucket": "qualification-scripts"}
    if region != "us-east-1":
        request["CreateBucketConfiguration"] = {"LocationConstraint": region}
    with Stubber(s3) as stub:
        if existing:
            stub.add_client_error("create_bucket", "BucketAlreadyOwnedByYou", expected_params=request)
        else:
            stub.add_response("create_bucket", {}, request)
        jobs.handle_script_upload(s3, "script.py", "scripts/job.py", "qualification_scripts")
        stub.assert_no_pending_responses()
    upload.assert_called_once_with("script.py", "qualification-scripts", "scripts/job.py")


def definition():
    return dict(name="features", description="qualification", record_identifier_feature="record",
                event_time_feature="time", feature_definitions=[], online_store_config={}, offline_store_config={},
                aws_access_key="testing", aws_secret_key="testing", region="us-west-2", role="qualification",
                data_lake_bucket="data", environment_prefix="qualification-")


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("code", ["AccessDenied", "BucketAlreadyExists", "InvalidLocationConstraint"])
def test_feature_storage_failure_prevents_delete_and_create(mocker, changed, code):
    sm = mocker.Mock()
    s3 = boto3.client("s3", region_name="us-west-2")
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, s3])
    mocker.patch.object(feature_store, "is_feature_group_new", return_value=not changed)
    mocker.patch.object(feature_store, "is_feature_group_different", return_value=changed)
    with Stubber(s3) as stub:
        stub.add_client_error("create_bucket", code, expected_params={
            "Bucket": "qualification-data", "CreateBucketConfiguration": {"LocationConstraint": "us-west-2"},
        })
        with pytest.raises(ClientError):
            feature_store.define_feature_group(**definition(), allow_recreate=changed)
        stub.assert_no_pending_responses()
    sm.delete_feature_group.assert_not_called()
    sm.create_feature_group.assert_not_called()


def test_recreate_confirms_bucket_then_delete_then_absence_then_create(mocker):
    events = []
    sm, s3 = mocker.Mock(), mocker.Mock()
    s3.meta = SimpleNamespace(region_name="us-east-1")
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, s3])
    mocker.patch.object(feature_store, "is_feature_group_new", return_value=False)
    mocker.patch.object(feature_store, "is_feature_group_different", return_value=True)
    s3.create_bucket.side_effect = lambda **kw: events.append("bucket")
    sm.delete_feature_group.side_effect = lambda **kw: events.append("delete")
    def absent(**kw):
        events.append("absent")
        raise error("ResourceNotFound", "DescribeFeatureGroup")
    sm.describe_feature_group.side_effect = absent
    sm.create_feature_group.side_effect = lambda **kw: events.append("create") or {}
    feature_store.define_feature_group(**definition(), allow_recreate=True)
    assert events == ["bucket", "delete", "absent", "create"]


def test_deletion_polling_waits_for_provider_not_found(mocker):
    sm = mocker.Mock()
    sm.describe_feature_group.side_effect = [
        {"FeatureGroupStatus": "Deleting"}, error("ResourceNotFound", "DescribeFeatureGroup"),
    ]
    pause = mocker.patch.object(feature_store, "sleep")
    feature_store.wait_for_feature_group_deletion(sm, "features", attempts=3, delay=1)
    assert sm.describe_feature_group.call_count == 2
    pause.assert_called_once_with(1)


@pytest.mark.parametrize("state,exception", [("DeleteFailed", RuntimeError), ("Deleting", TimeoutError)])
def test_failed_or_unconfirmed_deletion_never_recreates(mocker, state, exception):
    sm, s3 = mocker.Mock(), mocker.Mock()
    s3.meta = SimpleNamespace(region_name="us-east-1")
    sm.describe_feature_group.return_value = {"FeatureGroupStatus": state}
    mocker.patch.object(feature_store.boto3, "client", side_effect=[sm, s3])
    mocker.patch.object(feature_store, "is_feature_group_new", return_value=False)
    mocker.patch.object(feature_store, "is_feature_group_different", return_value=True)
    mocker.patch.object(feature_store, "sleep")
    with pytest.raises(exception):
        feature_store.define_feature_group(**definition(), allow_recreate=True)
    sm.create_feature_group.assert_not_called()
    assert sm.describe_feature_group.call_count == (1 if state == "DeleteFailed" else 120)


def test_deletion_polling_propagates_access_denial(mocker):
    sm = mocker.Mock()
    sm.describe_feature_group.side_effect = error("AccessDeniedException", "DescribeFeatureGroup")
    pause = mocker.patch.object(feature_store, "sleep")
    with pytest.raises(ClientError):
        feature_store.wait_for_feature_group_deletion(sm, "features")
    pause.assert_not_called()


@pytest.mark.parametrize("error_metric,expected", [(False, "high"), (True, "low")])
def test_winner_maximizes_score_and_minimizes_error(mocker, error_metric, expected):
    sm = mocker.Mock()
    sm.list_trial_components.return_value = {"TrialComponentSummaries": [
        {"TrialComponentName": "aws-training-job-" + name, "Status": {"PrimaryStatus": "Completed"}}
        for name in ["low", "high"]
    ]}
    details = [{"Status": {"PrimaryStatus": "Completed"}, "Source": {"SourceArn": name},
                "Metrics": [{"MetricName": "validation:score", "Max": value, "SourceArn": name}],
                "OutputArtifacts": {"SageMaker.ModelArtifact": {"Value": "s3://qualification/" + name}}}
               for name, value in [("low", 0.1), ("high", 0.9)]]
    sm.describe_trial_component.side_effect = details
    mocker.patch.object(experiment.time, "sleep")
    winner, model = experiment.detect_experiment_winner(sm, "qualification", "validation:score", "Max", error_metric, False)
    assert winner["Source"]["SourceArn"] == expected
    assert model == "s3://qualification/" + expected


def test_tracking_cleanup_failure_preserves_original_body_exception(mocker):
    client = mocker.Mock()
    client.update_trial_component.side_effect = error("AccessDeniedException", "UpdateTrialComponent")
    original = ValueError("private-body-detail")
    context = _tracking.ParameterTracker(display_name="qualification", artifact_bucket="qualification",
                                        artifact_prefix="run", sagemaker_boto_client=client)
    with pytest.raises(ValueError) as caught:
        with context:
            raise original
    assert caught.value is original
    assert original.__notes__ == ["Tracking status update also failed; remote completion is unconfirmed"]
    assert client.update_trial_component.call_args.kwargs["Status"] == {
        "PrimaryStatus": "Failed", "Message": "Tracking context failed",
    }
    with pytest.raises(RuntimeError):
        context.log_parameters({"after": 1})


def test_tracking_completion_failure_is_not_reported_as_success(mocker):
    client = mocker.Mock()
    failure = error("AccessDeniedException", "UpdateTrialComponent")
    client.update_trial_component.side_effect = failure
    with pytest.raises(ClientError) as caught:
        with _tracking.ParameterTracker(display_name="qualification", artifact_bucket="qualification",
                                       artifact_prefix="run", sagemaker_boto_client=client):
            pass
    assert caught.value is failure


def test_child_process_and_shell_remain_denied(tmp_path):
    marker = tmp_path / "must-not-exist"
    with pytest.raises(PermissionError, match="Offline qualification"):
        subprocess.run([sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"], check=True)
    with pytest.raises(PermissionError, match="Offline qualification"):
        os.system("echo must-not-run")
    assert not marker.exists()


def test_default_sdk_credentials_preserve_standard_session_token(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SECRET_KEY", raising=False)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "dummy-session-token")
    assert resolve_aws_credentials() == (None, None)
    assert prepare_pipeline_variables() == ["qualification", None, None, "us-east-1"]
    key, secret = resolve_aws_credentials()
    session = boto3.Session(aws_access_key_id=key, aws_secret_access_key=secret, region_name="us-east-1")
    creds = session.get_credentials().get_frozen_credentials()
    assert (creds.access_key, creds.secret_key, creds.token) == ("testing", "testing", "dummy-session-token")


@pytest.mark.parametrize("explicit", [("key", None), (None, "secret"), ("", "secret"), ("key", "")])
def test_partial_explicit_credentials_never_mix_with_legacy_environment(explicit):
    with pytest.raises(ValueError, match="complete nonempty"):
        resolve_aws_credentials(*explicit)


@pytest.mark.parametrize("missing", ["AWS_ACCESS_KEY", "AWS_SECRET_KEY"])
def test_partial_legacy_credentials_are_not_accepted(monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="complete nonempty"):
        resolve_aws_credentials()


def test_default_credentials_reach_real_feature_store_sdk_without_legacy_pair(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SECRET_KEY", raising=False)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "dummy-session-token")
    group = feature_store.load_feature_group("features", region="us-east-1", environment_prefix="qualification-")
    creds = group.sagemaker_session.boto_session.get_credentials().get_frozen_credentials()
    assert creds.token == "dummy-session-token"
    assert group.name == "qualification-features"
