import boto3


def get_features_from_store(
    name,
    y_column=None,
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    role=None,
    environment_prefix=None,
):
    """
    Gets the features from the targeted feature store and extracts the label column
    if specified. Returns X, y values for training a model.
    """
    raise Exception("Not implemented")

    X = []
    y = []

    return [X, y]
