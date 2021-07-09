import argparse
import os.path
import re

from azureml.core import ComputeTarget
from azureml.core import Dataset
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import BanditPolicy
from azureml.train.hyperdrive import HyperDriveConfig
from azureml.train.hyperdrive import PrimaryMetricGoal
from azureml.train.hyperdrive import RandomParameterSampling
from azureml.train.hyperdrive import uniform

from azml.util import get_environment
from azml.util import get_workspace


def get_compute_target(ws, name):
    compute_target = ComputeTarget(ws, name)
    return compute_target


def get_experiment(ws, name):
    experiment = Experiment(ws, name)
    return experiment


def get_dataset(ws, name, version):
    dataset = Dataset.get_by_name(ws, name, version)
    return dataset


def sanitize(name):
    return re.sub("^[0-9]+|[^0-9a-zA-Z]+", "_", name)


def get_dataset_reference(dataset, local_run):
    if local_run:
        dst = f"/data/{dataset.name}/{dataset.version}/"
        if not os.path.exists(dst):
            dataset.download(dst)
        return ["--data-dir", dst, "--input", dataset.as_named_input(sanitize(dataset.name))]
    else:
        return ["--data-dir", dataset.as_download()]


def get_run_config(compute_target, env, dataset):
    ref = get_dataset_reference(dataset, compute_target == "local")
    src = ScriptRunConfig(
        source_directory="src", script="birds/train_model.py",
        arguments=["--num-epochs", 30, "--output-dir", "./outputs"] + ref,
        compute_target=compute_target, environment=env)
    return src


def get_hyperdrive_config(run_config):
    param_sampling = RandomParameterSampling({
            "learning-rate": uniform(0.0005, 0.005),
            "momentum": uniform(0.9, 0.99)
        }
    )
    early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)
    hyperdrive_config = HyperDriveConfig(
        run_config=run_config, hyperparameter_sampling=param_sampling, policy=early_termination_policy,
        primary_metric_name="best_val_acc", primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=8, max_concurrent_runs=4)
    return hyperdrive_config


def submit_experiment(experiment, config, wait_for_completion, show_output):
    run = experiment.submit(config)
    if wait_for_completion:
        run.wait_for_completion(show_output=show_output)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="birds-training")
    parser.add_argument("--dataset-version", type=str, default="latest")
    parser.add_argument("--compute-target", type=str, default="ml-compute-t4")
    parser.add_argument("--experiment-name", type=str, default="pytorch-birds-training")
    parser.add_argument("--hyperdrive", action="store_true")
    parser.add_argument("--wait-for-completion", action="store_true")
    parser.add_argument("--show-output", action="store_true")
    args = parser.parse_args()

    ws = get_workspace()
    local_run = args.compute_target == "local"
    compute_target = get_compute_target(ws, args.compute_target) if not local_run else args.compute_target
    env = get_environment(local_run)
    dataset = get_dataset(ws, args.dataset_name, args.dataset_version)
    run_config = get_run_config(compute_target, env, dataset)

    if args.hyperdrive:
        run_config = get_hyperdrive_config(run_config)

    experiment = get_experiment(ws, args.experiment_name)
    run = submit_experiment(experiment, run_config, args.wait_for_completion, args.show_output)
    print(run.id)


if __name__ == "__main__":
    main()
