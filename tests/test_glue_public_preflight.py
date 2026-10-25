"""Public entrypoint must reject known-invalid intent/state before all writes."""
from copy import deepcopy

import boto3
from botocore.exceptions import ClientError, ParamValidationError
from botocore.stub import Stubber
import pytest

from cyint_aws_ml_ops_tools.glue import jobs


def arguments(**overrides):
    values = dict(name="job", command="glueetl", file_location="script.py", destination_key="script.py",
                  trigger_name="trigger", trigger_type="ON_DEMAND", aws_access_key="testing",
                  aws_secret_key="testing", region="us-east-1", role="qualification",
                  glue_script_bucket="scripts", environment_prefix="qualification-", job_configuration={})
    return {**values, **overrides}


def clients(mocker):
    glue, s3 = mocker.Mock(), mocker.Mock()
    glue.get_trigger.return_value = {"Trigger": {"Type": "ON_DEMAND", "State": "CREATED"}}
    mocker.patch.object(jobs.boto3, "client", side_effect=[glue, s3])
    return glue, s3


def assert_no_writes(glue, s3):
    assert s3.mock_calls == []
    for method in ("create_job", "update_job", "create_trigger", "update_trigger"):
        getattr(glue, method).assert_not_called()


def test_public_trigger_conflict_stops_before_bucket_script_and_job_writes(mocker):
    glue, s3 = clients(mocker)
    glue.get_trigger.return_value = {"Trigger": {"Type": "SCHEDULED", "State": "CREATED"}}
    with pytest.raises(ValueError, match="explicit migration"):
        jobs.define_job(**arguments())
    glue.get_trigger.assert_called_once_with(Name="qualification-trigger")
    assert_no_writes(glue, s3)


@pytest.mark.parametrize("code", ["AccessDeniedException", "ThrottlingException", "ValidationException"])
def test_public_trigger_read_failure_stops_before_any_write(mocker, code):
    glue, s3 = clients(mocker)
    glue.get_trigger.side_effect = ClientError({"Error": {"Code": code}}, "GetTrigger")
    with pytest.raises(ClientError):
        jobs.define_job(**arguments())
    assert_no_writes(glue, s3)


@pytest.mark.parametrize("code", ["AccessDeniedException", "ThrottlingException"])
def test_public_job_read_failure_stops_before_any_write(mocker, code):
    glue, s3 = clients(mocker)
    glue.get_job.side_effect = ClientError({"Error": {"Code": code}}, "GetJob")
    with pytest.raises(ClientError):
        jobs.define_job(**arguments())
    glue.get_trigger.assert_not_called()
    assert_no_writes(glue, s3)


@pytest.mark.parametrize("kind,definition", [
    ("SCHEDULED", {"Schedule": 42}),
    ("CONDITIONAL", {"Predicate": {"Conditions": "invalid"}}),
    ("EVENT", {"EventBatchingCondition": {"BatchSize": "two"}}),
    ("ON_DEMAND", {"Actions": [{"JobName": 42}]}),
])
def test_real_sdk_trigger_shape_rejected_before_any_provider_lookup_or_write(mocker, kind, definition):
    glue, s3 = clients(mocker)
    with pytest.raises(ParamValidationError):
        jobs.define_job(**arguments(trigger_type=kind, trigger_definition=definition))
    assert glue.mock_calls == []
    assert s3.mock_calls == []


@pytest.mark.parametrize("configuration", [
    {"MaxRetries": "invalid"}, {"DefaultArguments": {"--name": 42}}, {"UnsupportedField": True},
])
def test_real_sdk_job_shape_rejected_before_any_provider_lookup_or_write(mocker, configuration):
    glue, s3 = clients(mocker)
    original = deepcopy(configuration)
    with pytest.raises(ParamValidationError):
        jobs.define_job(**arguments(job_configuration=configuration))
    assert glue.mock_calls == []
    assert s3.mock_calls == []
    assert configuration == original


def test_real_sdk_full_public_sequence_reads_before_writes_and_preserves_configuration(mocker):
    glue = boto3.client("glue", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")
    order = []
    glue.meta.events.register("before-parameter-build.*.*", lambda model, **kw: order.append(model.name))
    s3.meta.events.register("before-parameter-build.*.*", lambda model, **kw: order.append(model.name))
    upload = mocker.patch.object(s3, "upload_file", side_effect=lambda *a, **kw: order.append("UploadFile"))
    mocker.patch.object(jobs.boto3, "client", side_effect=[glue, s3])
    configuration = {"DefaultArguments": {"--qualification": "true"}, "MaxRetries": 1}
    original = deepcopy(configuration)
    command = {"Name": "glueetl", "ScriptLocation": "s3://qualification-scripts/script.py", "PythonVersion": "3"}
    action = [{"JobName": "qualification-job"}]
    with Stubber(glue) as g, Stubber(s3) as storage:
        g.add_client_error("get_job", "EntityNotFoundException", expected_params={"JobName": "qualification-job"})
        g.add_client_error("get_trigger", "EntityNotFoundException", expected_params={"Name": "qualification-trigger"})
        storage.add_response("create_bucket", {}, {"Bucket": "qualification-scripts"})
        g.add_response("create_job", {}, {"Name": "qualification-job", "Role": "qualification", "Command": command})
        g.add_response("update_job", {}, {"JobName": "qualification-job", "JobUpdate": {**original, "Role": "qualification", "Command": command}})
        g.add_response("create_trigger", {}, {"Name": "qualification-trigger", "Type": "ON_DEMAND", "Actions": action})
        g.add_response("update_trigger", {}, {"Name": "qualification-trigger", "TriggerUpdate": {"Actions": action}})
        result = jobs.define_job(**arguments(job_configuration=configuration))
        g.assert_no_pending_responses()
        storage.assert_no_pending_responses()
    assert order == ["GetJob", "GetTrigger", "CreateBucket", "UploadFile", "CreateJob", "UpdateJob", "CreateTrigger", "UpdateTrigger"]
    assert result["job"] == {} and result["trigger"] == {}
    upload.assert_called_once_with("script.py", "qualification-scripts", "script.py")
    assert configuration == original
