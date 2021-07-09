import argparse

from azureml.core import Model
from azureml.core.webservice import AciWebservice

from azml.util import get_workspace


def get_model(ws, model_name, model_version):
    if model_version == "latest":
        models = [m for m in Model.list(ws, name=model_name, latest=True)]
    else:
        models = [m for m in Model.list(ws, name=model_name) if str(m.version) == model_version]
    return models[0] if len(models) > 0 else None


if __name__ == "__main__":
    # TODO implement this :)
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-name", type=str, default="birds")
    parser.add_argument("--model-name", type=str, default="birds")
    parser.add_argument("--model-version", type=str, default="latest")
    args = parser.parse_args()

    ws = get_workspace()
    model = get_model(ws, args.model_name, args.model_version)

    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

    service = model.deploy(ws, args.service_name, [model], aci_config, overwrite=True)
    service.wait_for_deployment(True)

    # service.run()
