"""Offline unit-test boundary: never use operator or CI cloud credentials."""
import os
import sys
from pathlib import Path

for name in tuple(os.environ):
    if name.startswith("AWS_"):
        del os.environ[name]
os.environ.update({
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_ACCESS_KEY": "testing",
    "AWS_SECRET_KEY": "testing",
    "AWS_REGION": "us-east-1",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_EC2_METADATA_DISABLED": "true",
    "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    "AWS_CONFIG_FILE": os.devnull,
    "ENVIRONMENT_PREFIX": "qualification",
    "SAGEMAKER_ADMIN_CONFIG_OVERRIDE": str(Path(__file__).parent / "tests/fixtures/sagemaker-config.yaml"),
    "SAGEMAKER_USER_CONFIG_OVERRIDE": str(Path(__file__).parent / "tests/fixtures/sagemaker-config.yaml"),
})


def _deny_external_effects(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"}:
        raise RuntimeError("Offline qualification forbids network and child processes")


sys.addaudithook(_deny_external_effects)
