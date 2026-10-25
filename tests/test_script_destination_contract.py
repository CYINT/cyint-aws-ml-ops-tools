"""Glue must execute the same bucket/key object that the library uploaded."""
from urllib.parse import unquote, urlsplit

import pytest

from cyint_aws_ml_ops_tools.glue import jobs


@pytest.mark.parametrize("destination,expected_key", [
    ("scripts/job.py", "scripts/job.py"),
    ("scripts/job with spaces.py", "scripts/job with spaces.py"),
    ("scripts/æ—¥æœ¬èªž.py", "scripts/æ—¥æœ¬èªž.py"),
    ("scripts/a?#%.py", "scripts/a?#%.py"),
    ("s3://qualification-scripts/scripts/job.py", "scripts/job.py"),
    ("s3://qualification-scripts/scripts/job%20with%20spaces.py", "scripts/job with spaces.py"),
    ("s3://qualification-scripts/scripts/literal%252F.py", "scripts/literal%2F.py"),
    ('scripts/a\tb.py', 'scripts/a\tb.py'),
    ('s3://qualification-scripts/scripts/a%09b.py', 'scripts/a\tb.py'),
    ('scripts/a\rb.py', 'scripts/a\rb.py'),
    ('s3://qualification-scripts/scripts/a%0Db.py', 'scripts/a\rb.py'),
    ('scripts/a\nb.py', 'scripts/a\nb.py'),
    ('s3://qualification-scripts/scripts/a%0Ab.py', 'scripts/a\nb.py'),
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
    's3://qualification-scripts/scripts/a\x00b.py',
    's3://qualification-\x00scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x01b.py',
    's3://qualification-\x01scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x02b.py',
    's3://qualification-\x02scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x03b.py',
    's3://qualification-\x03scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x04b.py',
    's3://qualification-\x04scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x05b.py',
    's3://qualification-\x05scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x06b.py',
    's3://qualification-\x06scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x07b.py',
    's3://qualification-\x07scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x08b.py',
    's3://qualification-\x08scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\tb.py',
    's3://qualification-\tscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\nb.py',
    's3://qualification-\nscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x0bb.py',
    's3://qualification-\x0bscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x0cb.py',
    's3://qualification-\x0cscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\rb.py',
    's3://qualification-\rscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x0eb.py',
    's3://qualification-\x0escripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x0fb.py',
    's3://qualification-\x0fscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x10b.py',
    's3://qualification-\x10scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x11b.py',
    's3://qualification-\x11scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x12b.py',
    's3://qualification-\x12scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x13b.py',
    's3://qualification-\x13scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x14b.py',
    's3://qualification-\x14scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x15b.py',
    's3://qualification-\x15scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x16b.py',
    's3://qualification-\x16scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x17b.py',
    's3://qualification-\x17scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x18b.py',
    's3://qualification-\x18scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x19b.py',
    's3://qualification-\x19scripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x1ab.py',
    's3://qualification-\x1ascripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x1bb.py',
    's3://qualification-\x1bscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x1cb.py',
    's3://qualification-\x1cscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x1db.py',
    's3://qualification-\x1dscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x1eb.py',
    's3://qualification-\x1escripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x1fb.py',
    's3://qualification-\x1fscripts/scripts/job.py',
    's3://qualification-scripts/scripts/a\x7fb.py',
    's3://qualification-\x7fscripts/scripts/job.py',
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
