"""Real Glue request validation and fail-closed whole-experiment promotion."""
from copy import deepcopy
from types import SimpleNamespace

import boto3
from botocore.stub import Stubber
import pytest

from cyint_aws_ml_ops_tools.glue import jobs
from cyint_aws_ml_ops_tools.experimentation import experiment


TRIGGERS = [
    ("ON_DEMAND", {}),
    ("SCHEDULED", {"Schedule": "cron(0 12 * * ? *)"}),
    ("CONDITIONAL", {"Predicate": {"Conditions": [{"JobName": "prior", "State": "SUCCEEDED"}]}}),
    ("EVENT", {"EventBatchingCondition": {"BatchSize": 2}}),
]


@pytest.mark.parametrize("kind,definition", TRIGGERS)
@pytest.mark.parametrize("existing", [False, True])
def test_real_glue_trigger_requests_preserve_custom_actions_and_only_applicable_fields(kind, definition, existing):
    client = boto3.client("glue", region_name="us-east-1")
    supplied = {**deepcopy(definition), "Actions": [{"JobName": "custom-job"}], "Description": "qualification"}
    original = deepcopy(supplied)
    with Stubber(client) as stub:
        if existing:
            stub.add_response("get_trigger", {"Trigger": {"Type": kind}}, {"Name": "qualification-trigger"})
        else:
            stub.add_client_error("get_trigger", "EntityNotFoundException", expected_params={"Name": "qualification-trigger"})
            stub.add_response("create_trigger", {}, {"Name": "qualification-trigger", "Type": kind, **original})
        stub.add_response("update_trigger", {}, {"Name": "qualification-trigger", "TriggerUpdate": original})
        jobs.handle_trigger(client, "default-job", "qualification-trigger", kind, supplied)
        stub.assert_no_pending_responses()
    assert supplied == original


@pytest.mark.parametrize("kind,definition", TRIGGERS)
def test_trigger_default_actions_do_not_mutate_caller(kind, definition):
    original = deepcopy(definition)
    prepared = jobs.prepare_trigger_definition("qualification-job", kind, definition)
    assert prepared["Actions"] == [{"JobName": "qualification-job"}]
    assert definition == original


@pytest.mark.parametrize("kind,definition", [
    ("UNKNOWN", {}), ("SCHEDULED", {}), ("CONDITIONAL", {}), ("EVENT", {}),
    ("ON_DEMAND", {"Schedule": None}), ("ON_DEMAND", {"Predicate": {}}),
    ("ON_DEMAND", {"Actions": []}), ("ON_DEMAND", {"Typo": "value"}),
])
def test_invalid_trigger_intent_stops_before_upload_or_job_mutation(mocker, kind, definition):
    glue, s3 = mocker.Mock(), mocker.Mock()
    mocker.patch.object(jobs.boto3, "client", side_effect=[glue, s3])
    with pytest.raises(ValueError):
        jobs.define_job("job", "glueetl", "script.py", "script.py", "trigger", kind,
                        trigger_definition=definition, aws_access_key="testing", aws_secret_key="testing",
                        region="us-east-1", role="qualification", glue_script_bucket="scripts", environment_prefix="qualification-")
    assert glue.mock_calls == []
    assert s3.mock_calls == []


def test_existing_trigger_type_mismatch_never_updates(mocker):
    client = mocker.Mock()
    client.get_trigger.return_value = {"Trigger": {"Type": "SCHEDULED"}}
    with pytest.raises(ValueError, match="explicit migration"):
        jobs.handle_trigger(client, "job", "trigger", "ON_DEMAND", {})
    client.update_trigger.assert_not_called()
    client.create_trigger.assert_not_called()


def summary(name, status="Completed"):
    return {"TrialComponentName": "aws-training-job-" + name, "Status": {"PrimaryStatus": status}}


def details(name, score=0.5, status="Completed"):
    return {"TrialComponentName": "aws-training-job-" + name, "Status": {"PrimaryStatus": status},
            "Source": {"SourceArn": name}, "Metrics": [{"MetricName": "accuracy", "Last": score, "SourceArn": name}],
            "OutputArtifacts": {"SageMaker.ModelArtifact": {"Value": "s3://qualification/" + name}}}


def promote(sm, s3, mocker, **kwargs):
    mocker.patch.object(experiment.boto3, "Session", return_value=SimpleNamespace(client=lambda name: {"sagemaker": sm, "s3": s3}[name]))
    return experiment.promote_winner("experiment", "run", "staging", "accuracy", role="qualification",
                                     environment_prefix="qualification", experiment_bucket="bucket", verbose=False, **kwargs)


@pytest.mark.parametrize("status", ["Stopped", "Failed", "unknown", None])
def test_noncompleted_state_never_prepares_or_overwrites_staging(mocker, status):
    sm, s3 = mocker.Mock(), mocker.Mock()
    sm.list_trial_components.return_value = {"TrialComponentSummaries": [summary("bad", status)]}
    with pytest.raises(ValueError, match="training states"):
        promote(sm, s3, mocker)
    sm.describe_trial_component.assert_not_called()
    assert s3.mock_calls == []


def test_later_page_failure_stops_promotion_before_describe_or_copy(mocker):
    sm, s3 = mocker.Mock(), mocker.Mock()
    sm.list_trial_components.side_effect = [
        {"TrialComponentSummaries": [summary("first")], "NextToken": "page2"},
        {"TrialComponentSummaries": [summary("bad", "Failed")]},
    ]
    with pytest.raises(ValueError):
        promote(sm, s3, mocker)
    assert sm.list_trial_components.call_args_list[1].kwargs == {"ExperimentName": "qualification-experiment-run", "NextToken": "page2"}
    sm.describe_trial_component.assert_not_called()
    assert s3.mock_calls == []


def test_later_page_better_candidate_wins(mocker):
    sm = mocker.Mock()
    sm.list_trial_components.side_effect = [
        {"TrialComponentSummaries": [summary("first")], "NextToken": "page2"},
        {"TrialComponentSummaries": [summary("better")]},
    ]
    sm.describe_trial_component.side_effect = [details("first", 0.2), details("better", 0.9)]
    winner, model = experiment.detect_experiment_winner(sm, "qualification", "accuracy", "Last", False, False)
    assert winner["Source"]["SourceArn"] == "better"
    assert model == "s3://qualification/better"


def test_completed_winner_preserves_real_s3_staging_requests(mocker):
    sm = mocker.Mock()
    sm.list_trial_components.return_value = {"TrialComponentSummaries": [summary("winner")]}
    selected = details("winner", 0.9)
    sm.describe_trial_component.return_value = selected
    s3 = boto3.client("s3", region_name="us-west-2")
    with Stubber(s3) as stub:
        stub.add_response("create_bucket", {}, {
            "Bucket": "qualification-staging", "CreateBucketConfiguration": {"LocationConstraint": "us-west-2"},
        })
        stub.add_response("copy_object", {}, {
            "Bucket": "qualification-staging", "Key": "staged_model.tar.gz", "CopySource": "qualification/winner",
        })
        assert promote(sm, s3, mocker, region="us-west-2") is selected
        stub.assert_no_pending_responses()


def test_each_poll_enumerates_all_pages_before_ranking(mocker):
    sm = mocker.Mock()
    sm.list_trial_components.side_effect = [
        {"TrialComponentSummaries": [summary("first")], "NextToken": "page2"},
        {"TrialComponentSummaries": [summary("later", "InProgress")]},
        {"TrialComponentSummaries": [summary("first")], "NextToken": "page2"},
        {"TrialComponentSummaries": [summary("later")]},
    ]
    sm.describe_trial_component.side_effect = [details("first", 0.2), details("later", 0.9)]
    pause = mocker.patch.object(experiment.time, "sleep")
    winner, _ = experiment.detect_experiment_winner(sm, "qualification", "accuracy", "Last", False, False, max_wait_attempts=2)
    assert winner["Source"]["SourceArn"] == "later"
    assert sm.list_trial_components.call_count == 4
    pause.assert_called_once_with(1)


@pytest.mark.parametrize("status", ["InProgress", "Stopping"])
def test_bounded_wait_exhaustion_never_stages_unconfirmed_model(mocker, status):
    sm, s3 = mocker.Mock(), mocker.Mock()
    sm.list_trial_components.return_value = {"TrialComponentSummaries": [summary("pending", status)]}
    pause = mocker.patch.object(experiment.time, "sleep")
    with pytest.raises(TimeoutError, match="promotion stopped"):
        promote(sm, s3, mocker, max_wait_attempts=2, wait_interval=0)
    assert sm.list_trial_components.call_count == 2
    pause.assert_called_once_with(0)
    assert s3.mock_calls == []


def test_repeated_pagination_token_cannot_silently_truncate_or_loop(mocker):
    sm = mocker.Mock()
    sm.list_trial_components.return_value = {"TrialComponentSummaries": [summary("first")], "NextToken": "repeat"}
    with pytest.raises(RuntimeError, match="pagination"):
        experiment.detect_experiment_winner(sm, "qualification", "accuracy", "Last", False, False)
    assert sm.list_trial_components.call_count == 2
    sm.describe_trial_component.assert_not_called()


@pytest.mark.parametrize("changed_state", ["InProgress", "Stopped", None])
def test_detail_status_must_confirm_completion_before_copy(mocker, changed_state):
    sm, s3 = mocker.Mock(), mocker.Mock()
    sm.list_trial_components.return_value = {"TrialComponentSummaries": [summary("first")]}
    sm.describe_trial_component.return_value = details("first", status=changed_state)
    with pytest.raises(ValueError, match="completion changed"):
        promote(sm, s3, mocker)
    assert s3.mock_calls == []


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_metric_cannot_be_promoted(mocker, score):
    sm, s3 = mocker.Mock(), mocker.Mock()
    sm.list_trial_components.return_value = {"TrialComponentSummaries": [summary("first")]}
    sm.describe_trial_component.return_value = details("first", score)
    with pytest.raises(ValueError, match="finite"):
        promote(sm, s3, mocker)
    assert s3.mock_calls == []
