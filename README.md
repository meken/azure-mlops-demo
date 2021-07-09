# azure-mlops-demo

A sample project to illustrate how to use Azure ML and PyTorch for image classification

> Work in progres...

## Getting Started

Make sure that there's a Python 3.8 (virtual) environment, if you don't have that you can run the following command to create for instance a Conda environment:

```shell
# If you prefer Conda
conda create -n azure-mlops-demo python=3.8 

# Or alternatively, just vanilla Python 3
python3 -m venv .venv
```

Before running the rest of the commands, make sure that you've activated the corect environment:

```shell
# Conda style
activate azure-mlops-demo

# Or vanilla Python 3 virtual environment
source .venv/bin/activate
```

Now you can install the project requirements as well as the development packages:

```shell
pip install -e .[azure,dev]
```

In order to get the sample data locally, run the following command, it will create a data folder with training and validation data:

```shell
python -m birds.download_data
```

## Day to day tasks

After activating the correct environment, to start the training in local environment, run the following command from the top level:

```shell
python -m birds.train_model
```

In order to run tests, with coverage, run the following command from the top level:

```shell
coverage run -m pytest
```

Linting through flake, note that configuration is in setup.cfg:

```shell
flake8
```

## Running things on Azure

Assuming that there's no [Azure Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace) created yet, you can run the following command to create one in an existing resource group.

> Note that at the time of this writing `ml` extension for az-cli is experimental and must be installed through the command line `az extension add --name ml`

```shell
RG=...  # set the name of the resource group
WS=mlw-mlops-demo
az ml workspace create -g $RG --workspace-name $WS
```

The workspace is now created using default settings. You can set specific parameters and configure accesibility (through private endpoints) using `yaml` files for configuration. See [here](https://github.com/Azure/azureml-examples/tree/main/cli) for examples.

Once the workspace is created you can create the additional resources. Let's start with the compute target. The command below creates an Azure ML compute cluster that has a minimum size of 0 and can scale to 2 nodes when needed. Similar to the workspace creation, you can control many of the settings through the `yaml` file approach.

```shell
ML_COMPUTE=ml-compute-t4
VM_TYPE=Standard_NC4as_T4_v3  # NVIDIA T4 gpu machine, requires quota
az ml compute create -g $RG -w $WS -n $ML_COMPUTE --type AmlCompute --size $VM_TYPE --max-instances 2
```

Next step is to create a versioned dataset for training purposes. For the sake of this example we'll upload the local copy of the data, but typically the data source would be already available on a for instance a blob store. This is a two step process, first you need to create a `datastore` pointing to a blob store or an RDBMS, and then create a `dataset` based on file path/pattern (for blob store) or SQL query (for an RDBMS).

Since we'll be uploading local data, and currently `az ml` cli command doesn't support it, we'll resort to the `yaml` approach. Create a new file `birds-training-dataset.yml` with the following contents.

```yml
$schema: https://azuremlschemas.azureedge.net/latest/asset.schema.json
name: birds-training
version: 1
local_path: data/
```

And now you can upload that to the default datastore of the workspace.

```shell
az ml data create -g $RG -w $WS -f birds-training-dataset.yml
```

We can now our first training job on Azure. But before we do that we need a way to indicate the default workspace environment for Azure ML related operations. In order to do that you need a local copy of the `config.json`. You can now either download the `config.json` from the portal, or generate it based on your current configuration.

```shell
SUB=`az account show --query id -o tsv`  # assuming a single subscription
FMT='{"subscription_id":"%s","resource_group":"%s","workspace_name":"%s"}'
printf "$FMT" $SUB $RG $WS > config.json
```

Now we can finally submit the training job to run on the previously configured Azure ML compute cluster.

```shell
python -m azml.run_experiment --compute-target $ML_COMPUTE
```

It's also possible to use `local` as `compute-target` in which case the training job will be running on the local machine but the metrics and logs will still be tracked on Azure.

The module `run_experiment.py` supports many other command line parameters, check those and provide different values if necessary.
