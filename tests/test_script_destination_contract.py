"""Glue must execute the same bucket/key object that the library uploaded."""
from urllib.parse import unquote, urlsplit

import pytest

from cyint_aws_ml_ops_tools.glue import jobs


@pytest.mark.parametrize("destination,expected_key", [
    ("scripts/job.py", "scripts/job.py"),
    ("scripts/job with spaces.py", "scripts/job with spaces.py"),
    ("scripts/日本語.py", "scripts/日本語.py"),
    ("scripts/a?#%.py", "scripts/a?#%.py"),
    ("s3://qualification-scripts/scripts/job.py", "scripts/job.py"),
    ("s3://qualification-scripts/scripts/job%20with%20spaces.py", "scripts/job with spaces.py"),
    ("s3://qualification-scripts/scripts/literal%252F.py", "scripts/literal%2F.py"),
])
def test_public_job_binds_uploaded_key_and_canonical_script_uri(mocker, destination, expected_key):
    glue, s3 = mocker.Mock(), mocker.Mock()
    glue.get_trigger.return_value = {"Trigger": {"Type": "ON_DEMAND"}}
    s3.meta.region_name = "us-east-1"
    mocker.patch.object(jobs.boto3, "client", side_effect=[glue, s3])
    jobs.define_job("job", "glueetl", "script.py", destination, "trigger", "ON_DEMAND",
                    aws_access_key="testing", aws_secret_key="testing", region="us-east-1", role="qualification",
                    glue_script_bucket="scripts", environment_prefix="qualification-")
    s3.upload_file.assert_called_once_with("script.py", "qualification-scripts", expected_key)
    location = glue.update_job.call_args.kwargs["JobUpdate"]["Command"]["ScriptLocation"]
    uri = urlsplit(location)
    assert uri.scheme == "s3" and uri.netloc == "qualification-scripts"
    assert unquote(uri.path[1:]) == expected_key
    assert not uri.query and not uri.fragment


@pytest.mark.parametrize("destination", [
    "", "x" * 1025, "s3://other-bucket/script.py", "s3://qualification-scripts/",
    "s3://qualification-scripts/script.py?versionId=unknown", "s3://qualification-scripts/script.py#fragment",
    "s3://qualification-scripts/scripts/bad%XZ.py",
])
def test_invalid_or_conflicting_script_destination_stops_before_lookup_or_write(mocker, destination):
    glue, s3 = mocker.Mock(), mocker.Mock()
    mocker.patch.object(jobs.boto3, "client", side_effect=[glue, s3])
    with pytest.raises(ValueError):
        jobs.define_job("job", "glueetl", "script.py", destination, "trigger", "ON_DEMAND",
                        aws_access_key="testing", aws_secret_key="testing", region="us-east-1", role="qualification",
                        glue_script_bucket="scripts", environment_prefix="qualification-")
    assert glue.mock_calls == []
    assert s3.mock_calls == []


def test_direct_upload_helper_interprets_same_bucket_s3_uri_consistently(mocker):
    storage = mocker.Mock()
    storage.meta.region_name = "us-east-1"
    jobs.handle_script_upload(storage, "script.py", "s3://qualification-scripts/scripts/job%20with%20spaces.py", "qualification_scripts")
    storage.upload_file.assert_called_once_with("script.py", "qualification-scripts", "scripts/job with spaces.py")
