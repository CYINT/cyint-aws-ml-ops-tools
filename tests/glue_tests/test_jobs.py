from types import SimpleNamespace
from cyint_aws_ml_ops_tools.glue.jobs import define_job


def test_jobs(mocker):
    client = mocker.patch("boto3.client").return_value
    client.meta = SimpleNamespace(region_name="us-east-1")
    client.get_trigger.return_value = {"Trigger": {"Type": "ON_DEMAND"}}
    result = define_job(
        "test",
        "test",
        "test",
        "test",
        "test",
        "ON_DEMAND",
        {},
        {},
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
    )

    assert set(result) == {"job", "script", "trigger"}
