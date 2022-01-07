import os

import boto3
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.serializers import CSVSerializer
from sagemaker.session import Session

from ..universal.pipeline import (
    create_s3bucket_if_not_exist,
    prepare_pipeline_variables,
    sanitize_bucket_name,
)


def define_inference_endpoint(
    name,
    image,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer(),
    initial_instance_count=1,
    instance_type="ml.c4.xlarge",
    artifact_filename="model.tar.gz",
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    role=None,
    environment_prefix=None,
):

    """
    Setup a SageMaker inference endpoint based on the environment
    """
    (
        environment_prefix_name,
        aws_access_key_id,
        aws_secret_key_id,
        region_name,
    ) = prepare_pipeline_variables(
        environment_prefix, aws_access_key, aws_secret_key, region
    )

    boto_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    sagemaker_client = boto_session.client(
        service_name="sagemaker", region_name=region_name
    )

    sm = Session(boto_session=boto_session, sagemaker_client=sagemaker_client)

    role_arn = os.environ["DEPLOYMENT_ROLE"] if role is None else role
    model_name = sanitize_bucket_name(f"{environment_prefix_name}-{name}")
    sm_model = Model(
        image=image,
        model_data=f"s3://{model_name}/{artifact_filename}",
        role=role_arn,
        sagemaker_session=sm,
    )
    endpoint_name = f"{environment_prefix_name}-{name}-endpoint"

    return sm_model.deploy(
        initial_instance_count=initial_instance_count,
        serializer=serializer,
        deserializer=deserializer,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
