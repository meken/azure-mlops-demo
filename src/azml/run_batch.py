import argparse

from azureml.core import ComputeTarget
from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Experiment
from azureml.core.compute import AmlCompute
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import ParallelRunConfig
from azureml.pipeline.steps import ParallelRunStep

from azml.util import get_environment
from azml.util import get_workspace


def get_compute_target(ws, name):
    # k80_cluster = ComputeTarget(workspace=ws, name="ml-compute-k80")
    # t4_cluster = ComputeTarget(workspace=ws, name="ml-compute-t4")
    # cpu_cluster = ComputeTarget(workspace=ws, name="ml-compute-ds5v2")
    return ComputeTarget(workspace=ws, name=name)


def get_process_count_per_node(ws, compute_target):
    available_vms = AmlCompute.supported_vmsizes(ws)
    vm_size = compute_target.vm_size
    res = filter(lambda x: x["name"].lower() == vm_size.lower(), available_vms)
    match = next(res, None)
    if match:
        gpus = match["gpus"]
        cpus = match["vCPUs"]
        return gpus if gpus > 0 else cpus
    else:
        return 1


def run_experiment(ws, experiment_name, compute_target):
    batch_data_set = Dataset.get_by_name(ws, "birds-scoring")

    env = get_environment()

    data_store = Datastore.get(ws, "mldata")
    output_dir = OutputFileDatasetConfig(name="inferences", destination=(data_store, "birds/inferences"))
    output_dir.register_on_complete("birds-inferences")

    max_nodes = compute_target.scale_settings.maximum_node_count
    max_processes = get_process_count_per_node(ws, compute_target)
    parallel_run_config = ParallelRunConfig(
        source_directory="src",
        entry_script="azml/batch_score.py",
        mini_batch_size=512,
        error_threshold=1,  # -1
        environment=env,
        output_action="summary_only",  # "append_row"
        process_count_per_node=max_processes,
        compute_target=compute_target,
        # run_invocation_timeout=600,
        node_count=max_nodes)

    parallelrun_step = ParallelRunStep(
        name="batch-score-birds",
        parallel_run_config=parallel_run_config,
        inputs=[batch_data_set.as_named_input("input_images").as_mount()],
        output=output_dir,
        arguments=["--model-name", "birds", "--output-dir", output_dir],
        allow_reuse=False
    )

    pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
    return Experiment(ws, experiment_name).submit(pipeline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-target", type=str, default="ml-compute-t4")
    parser.add_argument("--experiment-name", type=str, default="pytorch-birds-batch")
    args = parser.parse_args()

    ws = get_workspace()
    compute_target = get_compute_target(ws, args.compute_target)
    run = run_experiment(ws, args.experiment_name, compute_target)
    print(run.id)


if __name__ == "__main__":
    main()
