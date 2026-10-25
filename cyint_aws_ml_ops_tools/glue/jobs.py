import os
import re
from urllib.parse import quote, unquote, urlsplit
from copy import deepcopy
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError
from botocore.session import get_session
from botocore.validate import validate_parameters

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

    # Pure request validation and current-state reads precede every cloud write.
    _, script_location = prepare_script_destination(glue_script_bucket_name.replace("_", "-"), destination_key)
    job_requests = prepare_job_requests(
        job_name, command, script_location, role_arn, python_version, job_configuration,
    )
    trigger_requests = prepare_trigger_requests(
        job_name, trigger_environment_name, trigger_type, trigger_definition,
    )
    new_job = is_new_job(glueclient, job_name)
    new_trigger = read_trigger_state(glueclient, trigger_environment_name, trigger_type)

    response_script = handle_script_upload(
        s3client, file_location, destination_key, glue_script_bucket_name,
    )
    response_job = apply_job_requests(glueclient, new_job, job_requests)
    response_trigger = apply_trigger_requests(glueclient, new_trigger, trigger_requests)
    return {"job": response_job, "script": response_script, "trigger": response_trigger}


@lru_cache(maxsize=1)
def glue_service_model():
    # Load the installed SDK request schema as data; no client/credentials/network.
    return get_session().get_service_model("glue")


def validate_glue_request(operation, request):
    validate_parameters(request, glue_service_model().operation_model(operation).input_shape)


def prepare_job_requests(name, command, destination_key, role_arn, python_version, job_configuration):
    if not isinstance(job_configuration, dict):
        raise TypeError("Job configuration must be a dictionary")
    command_object = {"Name": command, "ScriptLocation": destination_key, "PythonVersion": python_version}
    configuration = deepcopy(job_configuration)
    configuration["Command"] = command_object
    configuration.setdefault("Role", role_arn)
    create = {"Name": name, "Role": role_arn, "Command": command_object}
    update = {"JobName": name, "JobUpdate": configuration}
    validate_glue_request("CreateJob", create)
    validate_glue_request("UpdateJob", update)
    return create, update


def apply_job_requests(glueclient, new_job, requests):
    create, update = requests
    if new_job:
        glueclient.create_job(**create)
    return glueclient.update_job(**update)


def handle_job(glueclient, new_job, name, command, destination_key, role_arn, python_version, job_configuration={}):
    """Validate both requests before creating/updating a Glue job."""
    requests = prepare_job_requests(name, command, destination_key, role_arn, python_version, job_configuration)
    return apply_job_requests(glueclient, new_job, requests)




def prepare_script_destination(bucket, destination):
    """Return one object key and its canonical URI; never silently change bucket."""
    if not isinstance(destination, str) or not destination:
        raise ValueError("Script destination must be a nonempty object key or S3 URI")
    key = destination
    if destination.lower().startswith("s3://"):
        uri = urlsplit(destination)
        if uri.netloc != bucket or uri.query or uri.fragment:
            raise ValueError("S3 script URI must use the configured bucket and no query or fragment")
        if re.search(r"%(?![0-9a-fA-F]{2})", uri.path):
            raise ValueError("S3 script URI contains invalid percent encoding")
        key = unquote(uri.path[1:], encoding="utf-8", errors="strict")
    if not key or len(key.encode("utf-8")) > 1024:
        raise ValueError("Script object key must contain 1 to 1024 UTF-8 bytes")
    return key, "s3://" + bucket + "/" + quote(key, safe="/")


def handle_script_upload(s3client, file_location, destination_key, glue_script_bucket):
    """
    Uploads the Glue Job script to the S3 bucket, and creates the necessary environment bucket if it doesn't exist.
    """

    glue_script_bucket_sanitized = glue_script_bucket.replace("_", "-")
    object_key, _ = prepare_script_destination(glue_script_bucket_sanitized, destination_key)

    create_s3bucket_if_not_exist(glue_script_bucket_sanitized, s3client)

    response = s3client.upload_file(
        file_location, glue_script_bucket_sanitized, object_key
    )

    return response


def prepare_trigger_definition(job_name, trigger_type, trigger_definition):
    """Validate type-specific intent without mutating caller configuration."""
    supported = {"ON_DEMAND", "SCHEDULED", "CONDITIONAL", "EVENT"}
    if trigger_type not in supported:
        raise ValueError("Unsupported Glue trigger type")
    if not isinstance(trigger_definition, dict):
        raise TypeError("Trigger definition must be a dictionary")
    definition = deepcopy(trigger_definition)
    allowed = {"Actions", "Description", "Schedule", "Predicate", "EventBatchingCondition"}
    if set(definition) - allowed:
        raise ValueError("Unsupported Glue trigger definition fields")
    for field, kind in {"Schedule": "SCHEDULED", "Predicate": "CONDITIONAL", "EventBatchingCondition": "EVENT"}.items():
        if field in definition and trigger_type != kind:
            raise ValueError("Trigger field does not apply to the requested type")
        if trigger_type == kind and not definition.get(field):
            raise ValueError("Trigger type requires a nonempty " + field)
    definition.setdefault("Actions", [{"JobName": job_name}])
    if not isinstance(definition["Actions"], list) or not definition["Actions"]:
        raise ValueError("Trigger requires nonempty actions")
    return definition


def prepare_trigger_requests(job_name, trigger_name, trigger_type, trigger_definition):
    definition = prepare_trigger_definition(job_name, trigger_type, trigger_definition)
    create = {"Name": trigger_name, "Type": trigger_type, **definition}
    update = {"Name": trigger_name, "TriggerUpdate": definition}
    validate_glue_request("CreateTrigger", create)
    validate_glue_request("UpdateTrigger", update)
    return create, update


def read_trigger_state(glueclient, trigger_name, trigger_type):
    try:
        existing = glueclient.get_trigger(Name=trigger_name)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "EntityNotFoundException":
            raise
        return True
    if existing.get("Trigger", {}).get("Type") != trigger_type:
        raise ValueError("Existing Glue trigger type does not match; explicit migration required")
    return False


def apply_trigger_requests(glueclient, new_trigger, requests):
    create, update = requests
    if new_trigger:
        glueclient.create_trigger(**create)
    return glueclient.update_trigger(**update)


def handle_trigger(glueclient, job_name, trigger_name, trigger_type, trigger_definition):
    """Validate complete requests and current trigger compatibility before writes."""
    requests = prepare_trigger_requests(job_name, trigger_name, trigger_type, trigger_definition)
    new_trigger = read_trigger_state(glueclient, trigger_name, trigger_type)
    return apply_trigger_requests(glueclient, new_trigger, requests)



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
