from cyint_aws_ml_ops_tools.glue.jobs import define_job


def test_jobs(mocker):
    mocker.patch("boto3.client")
    result = define_job(
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
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
