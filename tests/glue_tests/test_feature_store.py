import sys

import pytest

sys.path.insert(0, "../../cyint_aws_ml_ops_tools")

from cyint_aws_ml_ops_tools.glue.feature_store import define_feature_group


def test_feature_store(mocker):
    mocker.patch("boto3.client")
    define_feature_group(
        "test",
        "test",
        "test",
        "test",
        [],
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
