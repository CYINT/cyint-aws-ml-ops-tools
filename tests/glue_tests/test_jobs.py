import sys

import pytest

sys.path.insert(0, "../../src")

from src.glue.jobs import define_job


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
