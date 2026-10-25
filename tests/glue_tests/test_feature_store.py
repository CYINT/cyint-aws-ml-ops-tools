from types import SimpleNamespace
from botocore.exceptions import ClientError
from cyint_aws_ml_ops_tools.glue.feature_store import define_feature_group


def test_feature_store_creates_new_group_without_deleting(mocker):
    sm, s3 = mocker.Mock(), mocker.Mock()
    s3.meta = SimpleNamespace(region_name="us-east-1")
    sm.describe_feature_group.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFound", "Message": "qualification"}}, "DescribeFeatureGroup"
    )
    mocker.patch("boto3.client", side_effect=[sm, s3])
    response = define_feature_group(
        "features", "qualification", "record", "time", [{"FeatureName": "record", "FeatureType": "String"}, {"FeatureName": "time", "FeatureType": "Fractional"}], {"EnableOnlineStore": True}, {}, None,
        "testing", "testing", "us-east-1", "arn:aws:iam::123456789012:role/qualification",
        "data", "qualification-",
    )
    sm.create_feature_group.assert_called_once()
    sm.delete_feature_group.assert_not_called()
    s3.create_bucket.assert_called_once_with(Bucket="qualification-data")
    assert response == {"feature_group": sm.create_feature_group.return_value}
