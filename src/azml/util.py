import configparser

from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.conda_dependencies import CondaDependencies


def get_workspace():
    cli_auth = AzureCliAuthentication()
    ws = Workspace.from_config(auth=cli_auth)
    return ws


def get_pip_packages():
    config = configparser.ConfigParser()
    config.read("setup.cfg")
    return config["options"]["install_requires"].split()


def get_environment(local=False):
    base_name = "pytorch-birds"

    if local:
        env = Environment(name=f"{base_name}-local")
        env.python.user_managed_dependencies = True
    else:
        pip_packages = get_pip_packages() + ["azureml-core", "azureml-mlflow", "azureml-dataprep[fuse]"]
        env = Environment(name=base_name)
        env.python.conda_dependencies = CondaDependencies.create(pip_packages=pip_packages, python_version="3.8")
        env.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04"

    return env
