import argparse

import torch

from azureml.core import Experiment
from azureml.core import Model

from azml.util import get_workspace


def get_experiment(ws, name):
    experiment = Experiment(ws, name)
    return experiment


def get_torch_version():
    ver = torch.__version__
    idx = ver.find("+")
    return ver[:idx] if idx >= 0 else ver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="pytorch-birds-resnet152")
    parser.add_argument("--run-id", type=str, required=False)
    args = parser.parse_args()

    ws = get_workspace()

    if args.run_id:
        run = ws.get_run(args.run_id)
    else:
        # try the latest run
        experiment = get_experiment(ws, args.experiment_name)
        runs = experiment.get_runs()
        run = next(runs, None)

    if run and run.status == "Completed":
        details = run.get_details()
        print(f"Registering model from run: {run.id}, completed on (UTC): {details['endTimeUtc']}")
        model = run.register_model(model_name="birds", model_path="outputs/model.pt",
                                   model_framework=Model.Framework.PYTORCH,
                                   model_framework_version=get_torch_version())
        print(f"Succesfully registered model {model.name}:{model.version}")
    else:
        print("No model to register")


if __name__ == "__main__":
    main()
