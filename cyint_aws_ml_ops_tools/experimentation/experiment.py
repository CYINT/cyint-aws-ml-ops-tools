import itertools
import math
import os
import time
from io import StringIO
from uuid import uuid4

import boto3
import sagemaker
from ._tracking import create_experiment, create_trial, ParameterTracker

from ..universal.pipeline import (
    create_s3bucket_if_not_exist,
    prepare_pipeline_variables,
    sanitize_bucket_name,
)


def define_experiment(
    name,
    hypothesis,
    model_training_function,
    metrics=[],
    static_hyperparams={},
    hyperparam_configuration={},
    dataset=[],
    k_fold_size=1,
    run=None,
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    role=None,
    environment_prefix=None,
    experiment_bucket=None,
    trial_wait_time=3,
):
    """
    Setup a SageMaker experiment based on the environment
    """
    (
        environment_prefix_name,
        aws_access_key_id,
        aws_secret_key_id,
        region_name,
    ) = prepare_pipeline_variables(
        environment_prefix, aws_access_key, aws_secret_key, region
    )

    experiment_bucket_override, role_arn = prepare_experiment_variables(
        experiment_bucket, role
    )

    run_id = run if run is not None else uuid4()
    experiment_name = generate_experiment_name(environment_prefix_name, name, run_id)
    experiment_bucket_name = generate_experiment_bucket_name(
        environment_prefix_name, experiment_bucket_override
    )
    sess = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    sm = sess.client("sagemaker")
    s3client = sess.client("s3")

    hyperparameter_set = define_hyperparameter_set(hyperparam_configuration)
    create_s3bucket_if_not_exist(experiment_bucket_name, s3client)

    training_experiment = create_experiment(
        experiment_name=experiment_name,
        description=hypothesis,
        sagemaker_boto_client=sm,
    )

    exp_tracker = track_experiment_metadata(
        training_experiment,
        experiment_bucket_name,
        name,
        run_id,
        static_hyperparams,
        hyperparam_configuration,
        sm,
    )

    create_trials(
        dataset,
        hyperparameter_set,
        static_hyperparams,
        training_experiment,
        experiment_bucket_name,
        name,
        run_id,
        metrics,
        exp_tracker,
        model_training_function,
        environment_prefix_name,
        sm,
        role_arn,
        trial_wait_time,
        k_fold_size,
        region_name,
    )

    return run_id


def promote_winner(
    experiment,
    run,
    staging_bucket,
    metric,
    error_metric=False,
    metric_value="Last",
    bonferroni_correction=True,
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
    role=None,
    environment_prefix=None,
    experiment_bucket=None,
    verbose=True,
    *,
    max_wait_attempts=3600,
    wait_interval=1,
):
    """
    Evaluate the results of an experiment run and promote the winning model into
    the next stage
    """
    (
        environment_prefix_name,
        aws_access_key_id,
        aws_secret_key_id,
        region_name,
    ) = prepare_pipeline_variables(
        environment_prefix, aws_access_key, aws_secret_key, region
    )

    experiment_bucket_override, role_arn = prepare_experiment_variables(
        experiment_bucket, role
    )

    experiment_name = generate_experiment_name(environment_prefix_name, experiment, run)
    staging_bucket_name = sanitize_bucket_name(
        f"{environment_prefix_name}-{staging_bucket}"
    )
    sess = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    sm = sess.client("sagemaker")
    s3client = sess.client("s3")

    winner, winning_model = detect_experiment_winner(
        sm, experiment_name, metric, metric_value, error_metric, verbose,
        max_wait_attempts=max_wait_attempts, wait_interval=wait_interval,
    )

    create_s3bucket_if_not_exist(staging_bucket_name, s3client)

    s3client.copy_object(
        Bucket=staging_bucket_name,
        Key="staged_model.tar.gz",
        CopySource=winning_model.replace("s3://", ""),
    )

    if verbose:
        print(f"Trial {winner['TrialComponentName']} promoted to staging.")

    return winner


def detect_experiment_winner(
    sm, experiment_name, metric, metric_value, error_metric, verbose=True,
    *, max_wait_attempts=3600, wait_interval=1,
):
    """Rank only a fully enumerated, successfully completed experiment."""
    if type(max_wait_attempts) is not int or max_wait_attempts < 1:
        raise ValueError("max_wait_attempts must be a positive integer")
    if not isinstance(wait_interval, (int, float)) or not math.isfinite(wait_interval) or wait_interval < 0:
        raise ValueError("wait_interval must be finite and nonnegative")
    for attempt in range(max_wait_attempts):
        summaries = []
        token = None
        seen = set()
        while True:
            request = {"ExperimentName": experiment_name}
            if token is not None:
                request["NextToken"] = token
            response = sm.list_trial_components(**request)
            summaries.extend(response["TrialComponentSummaries"])
            token = response.get("NextToken")
            if not token:
                break
            if not isinstance(token, str) or token in seen or len(seen) >= 1000:
                raise RuntimeError("Trial component pagination did not complete")
            seen.add(token)
        training_jobs = [job for job in summaries if "aws-training-job" in job["TrialComponentName"]]
        if not training_jobs:
            raise ValueError("No training jobs found in this experiment")
        names = [job["TrialComponentName"] for job in training_jobs]
        if len(names) != len(set(names)):
            raise RuntimeError("Trial component enumeration returned duplicate training jobs")
        states = [job.get("Status", {}).get("PrimaryStatus") for job in training_jobs]
        if any(state not in {"Completed", "InProgress", "Stopping"} for state in states):
            raise ValueError("Experiment contains failed, stopped or unknown training states")
        if all(state == "Completed" for state in states):
            break
        if attempt + 1 == max_wait_attempts:
            raise TimeoutError("Experiment completion was not confirmed; promotion stopped")
        if verbose:
            print(".", end="")
        time.sleep(wait_interval)
    if verbose:
        print("")
    candidates = []
    for job in training_jobs:
        details = sm.describe_trial_component(TrialComponentName=job["TrialComponentName"])
        if details.get("Status", {}).get("PrimaryStatus") != "Completed":
            raise ValueError("Training component completion changed or is unconfirmed")
        matches = [entry for entry in details["Metrics"] if metric in entry["MetricName"]]
        if len(matches) != 1:
            raise ValueError("Winner metric must identify exactly one metric per training component")
        value = float(matches[0][metric_value])
        if not math.isfinite(value):
            raise ValueError("Winner metric must be finite")
        candidates.append((value, details))
    candidates.sort(key=lambda item: item[0], reverse=not error_metric)
    winner = candidates[0][1]
    winning_model = winner["OutputArtifacts"]["SageMaker.ModelArtifact"]["Value"]
    return [winner, winning_model]



def define_hyperparameter_set(hyperparam_configuration):
    # The Cartesian product of an empty grid is one static-only trial.
    hypnames = tuple(hyperparam_configuration)
    hypvalues = tuple(hyperparam_configuration.values())
    trial_hyperparameter_set = [
        dict(zip(hypnames, h)) for h in itertools.product(*hypvalues)
    ]
    return trial_hyperparameter_set


def track_experiment_metadata(
    training_experiment,
    experiment_bucket,
    name,
    run_id,
    static_hyperparams,
    hyperparam_configuration,
    sm,
):
    with ParameterTracker(
        display_name=training_experiment.experiment_name,
        artifact_bucket=experiment_bucket,
        artifact_prefix=f"{name}/{run_id}",
        sagemaker_boto_client=sm,
    ) as exp_tracker:
        exp_tracker.log_parameters(static_hyperparams)
        exp_tracker.log_parameters(hyperparam_configuration)
        # exp_tracker.log_artifact(file_path=f"{experiment_bucket}/{name}/{run_id}")

    return exp_tracker


def create_trials(
    dataset,
    trial_hyperparameter_set,
    static_hyperparams,
    training_experiment,
    experiment_bucket,
    name,
    run_id,
    metrics,
    exp_tracker,
    model_training_function,
    environment_prefix_name,
    sm,
    role,
    wait_time,
    kfold_size,
    region_name,
):
    for data_index, data in enumerate(dataset):
        for hyp_index, trial_hyp in enumerate(trial_hyperparameter_set):
            # Combine static hyperparameters and trial specific hyperparameters
            hyperparams = {**static_hyperparams, **trial_hyp}

            # Create unique job name with hyperparameter and time
            time_append = int(time.time())
            hyp_append = f"-dataset-{data_index}-hparams-{hyp_index}"
            job_name = f"{environment_prefix_name.replace('_','-')}-{name.replace('_','-')}-{str(run_id)[:8]}{hyp_append}-{time_append}"

            # Create a Tracker to track Trial specific hyperparameters
            with ParameterTracker(
                display_name=f"trial-metadata-{time_append}",
                artifact_bucket=experiment_bucket,
                artifact_prefix=f"{name}/{run_id}",
                sagemaker_boto_client=sm,
            ) as trial_tracker:
                trial_tracker.log_parameters(hyperparams)

            # Create a new Trial and associate Tracker to it
            tf_trial = create_trial(
                trial_name=f"trial{hyp_append}-{time_append}",
                experiment_name=training_experiment.experiment_name,
                sagemaker_boto_client=sm,
            )
            tf_trial.add_trial_component(exp_tracker.trial_component)
            time.sleep(wait_time)  # To prevent ThrottlingException
            tf_trial.add_trial_component(trial_tracker.trial_component)

            # Create an experiment config that associates training job to the Trial
            experiment_config = {
                "ExperimentName": training_experiment.experiment_name,
                "TrialName": tf_trial.trial_name,
                "TrialComponentDisplayName": job_name,
            }

            model_training_function(
                data,
                f"s3://{experiment_bucket}/{name}/{run_id}/{tf_trial.trial_name}",
                f"s3://{experiment_bucket}/code",
                role,
                metrics,
                sm,
                hyperparams,
                job_name,
                experiment_config,
                experiment_bucket,
                kfold_size,
                region_name,
            )

            time.sleep(wait_time)  # To prevent ThrottlingException


def upload_to_s3_pipeline(
    object,
    key,
    experiment_bucket=None,
    environment_prefix=None,
    aws_access_key=None,
    aws_secret_key=None,
    region=None,
):
    (
        environment_prefix_name,
        aws_access_key_id,
        aws_secret_key_id,
        region_name,
    ) = prepare_pipeline_variables(
        environment_prefix, aws_access_key, aws_secret_key, region
    )

    experiment_bucket_override = (
        os.environ["EXPERIMENT_BUCKET"]
        if experiment_bucket is None
        else experiment_bucket
    )

    experiment_bucket_name = f"{environment_prefix_name}-{experiment_bucket_override}"
    sess = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_key_id,
        region_name=region_name,
    )

    s3client = sess.client("s3")
    s3client.put_object(Bucket=experiment_bucket_name, Body=object, Key=key)


def prepare_experiment_variables(experiment_bucket, role):
    experiment_bucket_override = (
        os.environ["EXPERIMENT_BUCKET"]
        if experiment_bucket is None
        else experiment_bucket
    )
    role_arn = os.environ["EXPERIMENTATIONS_ROLE"] if role is None else role
    return [experiment_bucket_override, role_arn]


def generate_experiment_name(environment_prefix_name, name, run_id):
    return (
        f"{environment_prefix_name.replace('_','-')}-{name.replace('_', '-')}-{run_id}"
    )


def generate_experiment_bucket_name(
    environment_prefix_name, experiment_bucket_override
):
    return f"{environment_prefix_name}-{experiment_bucket_override}"
