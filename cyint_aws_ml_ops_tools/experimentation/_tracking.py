"""Parameter-only experiment tracking through the public SageMaker boto3 API.

These records implement the operations used by this library, not the legacy
smexperiments SDK. They never load cloud credentials or create service clients.
The caller supplies the same client used for its experiment and training jobs.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from numbers import Real
from uuid import uuid4


@dataclass(frozen=True)
class ExperimentReference:
    experiment_name: str


@dataclass(frozen=True)
class TrialComponentReference:
    trial_component_name: str


@dataclass(frozen=True)
class TrialReference:
    trial_name: str
    _client: object = field(repr=False, compare=False)

    def add_trial_component(self, component):
        name = component if isinstance(component, str) else component.trial_component_name
        return self._client.associate_trial_component(
            TrialName=self.trial_name, TrialComponentName=name,
        )


def create_experiment(*, experiment_name, description, sagemaker_boto_client):
    sagemaker_boto_client.create_experiment(
        ExperimentName=experiment_name, Description=description,
    )
    return ExperimentReference(experiment_name)


def create_trial(*, trial_name, experiment_name, sagemaker_boto_client):
    sagemaker_boto_client.create_trial(
        TrialName=trial_name, ExperimentName=experiment_name,
    )
    return TrialReference(trial_name, sagemaker_boto_client)


def _parameter_value(value):
    if isinstance(value, str):
        return {"StringValue": value}
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Tracking parameters must be finite")
        return {"NumberValue": numeric}
    if value is None or isinstance(value, (bool, list, tuple, dict)):
        return {"StringValue": json.dumps(value, sort_keys=True, allow_nan=False)}
    raise TypeError("Unsupported tracking parameter value")


class ParameterTracker:
    """Track parameters and lifecycle; artifact upload is outside this class."""

    def __init__(self, *, display_name, artifact_bucket, artifact_prefix, sagemaker_boto_client):
        self._client = sagemaker_boto_client
        self._display_name = display_name
        self.artifact_bucket = artifact_bucket
        self.artifact_prefix = artifact_prefix
        self.trial_component = TrialComponentReference("cyint-metadata-" + uuid4().hex)
        self._parameters = {}
        self._active = False
        self._used = False

    def __enter__(self):
        if self._used:
            raise RuntimeError("A parameter tracker context cannot be reused")
        self._used = True
        self._client.create_trial_component(
            TrialComponentName=self.trial_component.trial_component_name,
            DisplayName=self._display_name,
            Status={"PrimaryStatus": "InProgress"},
            StartTime=datetime.now(timezone.utc),
        )
        self._active = True
        return self

    def log_parameters(self, parameters):
        if not self._active:
            raise RuntimeError("Parameters must be logged inside their tracking context")
        converted = {}
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Tracking parameter names must be nonempty strings")
            converted[name] = _parameter_value(value)
        merged = {**self._parameters, **converted}
        if len(merged) > 300:
            raise ValueError("SageMaker supports at most 300 tracking parameters")
        self._parameters = merged

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self._active:
            raise RuntimeError("The tracking context is not active")
        self._active = False
        status = {"PrimaryStatus": "Completed"}
        if exc_type is not None:
            # Never copy exception text, which can include credentials or data, to cloud metadata.
            status = {"PrimaryStatus": "Failed", "Message": "Tracking context failed"}
        self._client.update_trial_component(
            TrialComponentName=self.trial_component.trial_component_name,
            Status=status, EndTime=datetime.now(timezone.utc), Parameters=self._parameters,
        )
        return False
