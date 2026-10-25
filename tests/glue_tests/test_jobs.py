from types import SimpleNamespace
from cyint_aws_ml_ops_tools.glue.jobs import define_job


def test_jobs(mocker):
    client = mocker.patch("boto3.client").return_value
    client.meta = SimpleNamespace(region_name="us-east-1")
    client.get_trigger.return_value = {"Trigger": {"Type": "ON_DEMAND", "State": "CREATED"}}
    result = define_job(
        "job", "glueetl", "script.py", "script.py", "trigger", "ON_DEMAND",
        aws_access_key="testing", aws_secret_key="testing", region="us-east-1",
        role="qualification", glue_script_bucket="scripts", environment_prefix="qualification-",
    )

    assert set(result) == {"job", "script", "trigger"}
