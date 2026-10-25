import os

import boto3
from botocore.exceptions import ClientError

from cyint_aws_ml_ops_tools.universal.pipeline import create_s3bucket_if_not_exist, resolve_aws_credentials


def define_job(
    name,
    command,
    file_location,
    destination_key,
    trigger_name,
    trigger_type,
    trigger_definition={},
    job_configuration={},
    python_version="3",
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    role=None,
    glue_script_bucket=None,
    environment_prefix=None,
):
    """
    Primary function to call when defining a new job in a Jupyter notebook or script. This function is designed to succeed regardless of existing
    state of the system.
    """
    environment_prefix_name = environment_prefix

    aws_access_key_id, aws_secret_key_id = resolve_aws_credentials(aws_access_key, aws_secret_key)
    region_name = region if region is not None else os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    role_arn = os.environ["GLUE_ROLE"] if role is None else role
    glue_script_bucket_name = (
        os.environ["GLUE_SCRIPT_BUCKET"]
        if glue_script_bucket is None
        else glue_script_bucket
    )
    environment_prefix_name = (
        os.environ["ENVIRONMENT_PREFIX"]
        if environment_prefix is None
        else environment_prefix
    )

    glueclient = boto3.client(
        "glue",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    s3client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    job_name = f"{environment_prefix_name}{name}"
    trigger_environment_name = f"{environment_prefix_name}{trigger_name}"
    glue_script_bucket_name = f"{environment_prefix_name}{glue_script_bucket_name}"

    response_script = handle_script_upload(
        s3client,
        file_location,
        destination_key,
        glue_script_bucket_name,
    )

    new_job = is_new_job(glueclient, job_name)
    response_job = handle_job(
        glueclient,
        new_job,
        job_name,
        command,
        destination_key,
        role_arn,
        python_version,
        job_configuration,
    )
    response_trigger = handle_trigger(
        glueclient, job_name, trigger_environment_name, trigger_type, trigger_definition
    )
    return {"job": response_job, "script": response_script, "trigger": response_trigger}


def handle_job(
    glueclient,
    new_job,
    name,
    command,
    destination_key,
    role_arn,
    python_version,
    job_configuration={},
):
    """
    Sets up the job in AWS Glue or updates it if it already exists.
    """

    command_object = {
        "Name": command,
        "ScriptLocation": destination_key,
        "PythonVersion": python_version,
    }

    if new_job:
        response = glueclient.create_job(
            Name=name,
            Role=role_arn,
            Command=command_object,
        )

    job_configuration["Command"] = command_object
    job_configuration["Role"] = (
        role_arn if "Role" not in job_configuration else job_configuration["Role"]
    )

    response = glueclient.update_job(JobName=name, JobUpdate=job_configuration)

    return response


def handle_script_upload(s3client, file_location, destination_key, glue_script_bucket):
    """
    Uploads the Glue Job script to the S3 bucket, and creates the necessary environment bucket if it doesn't exist.
    """

    glue_script_bucket_sanitized = glue_script_bucket.replace("_", "-")

    create_s3bucket_if_not_exist(glue_script_bucket_sanitized, s3client)

    response = s3client.upload_file(
        file_location, glue_script_bucket_sanitized, destination_key
    )

    return response


def handle_trigger(
    glueclient, job_name, trigger_name, trigger_type, trigger_definition
):
    """
    Sets up a trigger in AWS Glue or updates it if it already exists.
    """
    new_trigger = is_new_trigger(glueclient, trigger_name)

    schedule = None if trigger_type != "SCHEDULED" else trigger_definition["Schedule"]
    predicate = {} if trigger_type != "CONDITIONAL" else trigger_definition["Predicate"]
    event_batching = (
        {"BatchSize": 1}
        if trigger_type != "EVENT"
        else trigger_definition["EventBatchingCondition"]
    )

    actions = (
        [{"JobName": job_name}]
        if "Actions" not in trigger_definition
        else trigger_definition["Actions"]
    )
    trigger_definition["Actions"] = actions

    if new_trigger:
        response = glueclient.create_trigger(
            Name=trigger_name,
            Type=trigger_type,
            Actions=[{"JobName": job_name}],
            Schedule=schedule,
            Predicate=predicate,
            EventBatchingCondition=event_batching,
        )

    response = glueclient.update_trigger(
        Name=trigger_name, TriggerUpdate=trigger_definition
    )

    return response


def is_new_job(glueclient, name):
    """
    Test to determine if the AWS Glue job name already exists.
    """
    try:
        glueclient.get_job(JobName=name)

        return False
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "EntityNotFoundException":
            return True
        raise


def is_new_trigger(glueclient, name):
    """
    Test to determine if the AWS Glue Trigger name already exists.
    """
    try:
        glueclient.get_trigger(Name=name)

        return False
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "EntityNotFoundException":
            return True
        raise
