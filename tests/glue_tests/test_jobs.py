import sys

import pytest

sys.path.insert(0, "../../cyint_aws_ml_ops_tools")

from cyint_aws_ml_ops_tools.glue.jobs import define_job


def test_jobs(mocker):
    mocker.patch("boto3.client")
    define_job(
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
        "test",
    )
