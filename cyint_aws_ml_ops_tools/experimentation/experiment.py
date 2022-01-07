import itertools
import os
import time
from io import StringIO
from uuid import uuid4

import boto3
import sagemaker
from smexperiments.experiment import Experiment
from smexperiments.tracker import Tracker
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent

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
    s3client.create_bucket(Bucket=experiment_bucket_name)

    try:
        s3client.create_bucket(Bucket=experiment_bucket_name)
    except ClientError as e:
        pass

    training_experiment = Experiment.create(
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
        sm, experiment_name, metric, metric_value, error_metric, verbose
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
    sm, experiment_name, metric, metric_value, error_metric, verbose=True
):
    inprogress_jobs = ["placeholder"]
    while len(inprogress_jobs) > 0:
        components = sm.list_trial_components(ExperimentName=experiment_name)

        training_jobs = list(
            filter(
                lambda x: "aws-training-job" in x["TrialComponentName"],
                components["TrialComponentSummaries"],
            )
        )
        if len(training_jobs) < 1:
            raise Exception("No training jobs found in this experiment.")

        inprogress_jobs = list(
            filter(
                lambda x: x["Status"]["PrimaryStatus"] == "InProgress", training_jobs
            )
        )
        if verbose:
            print(".", end="")
        time.sleep(1)
    if verbose:
        print("")
    failed_jobs = list(
        filter(lambda x: x["Status"]["PrimaryStatus"] == "Failed", training_jobs)
    )

    if len(failed_jobs) > 0:
        raise Exception(
            [
                {"trial": job["TrialComponentName"], "Reason": job["Status"]["Message"]}
                for job in failed_jobs
            ]
        )

    job_dictionary = {}
    metrics = []

    for job in training_jobs:
        job_details = sm.describe_trial_component(
            TrialComponentName=job["TrialComponentName"]
        )
        job_dictionary[job_details["Source"]["SourceArn"]] = job_details
        metrics.append(
            list(filter(lambda x: metric in x["MetricName"], job_details["Metrics"]))[0]
        )
    metrics.sort(key=lambda x: float(x[metric_value]), reverse=error_metric)
    winner = job_dictionary[metrics[0]["SourceArn"]]
    winning_model = winner["OutputArtifacts"]["SageMaker.ModelArtifact"]["Value"]

    return [winner, winning_model]


def define_hyperparameter_set(hyperparam_configuration):
    hypnames, hypvalues = zip(*hyperparam_configuration.items())
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
    with Tracker.create(
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
            with Tracker.create(
                display_name=f"trial-metadata-{time_append}",
                artifact_bucket=experiment_bucket,
                artifact_prefix=f"{name}/{run_id}",
                sagemaker_boto_client=sm,
            ) as trial_tracker:
                trial_tracker.log_parameters(hyperparams)

            # Create a new Trial and associate Tracker to it
            tf_trial = Trial.create(
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
