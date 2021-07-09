import argparse
import datetime
import os
import uuid

from azureml.core import Model

from birds import datasets
from birds import score_model


def init():
    global model, data_transforms, output_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--output-dir", type=str)
    args, _ = parser.parse_known_args()

    model_path = Model.get_model_path(model_name=args.model_name)
    model = score_model.get_model(model_path)
    data_transforms = score_model.get_data_transforms()
    output_dir = f"{args.output_dir}/{str(datetime.date.today())}"
    os.makedirs(output_dir, exist_ok=True)


def run(mini_batch):
    dataset = datasets.FilePathDataset(mini_batch, data_transforms)
    res = score_model.run(model, dataset)
    with open(f"{output_dir}/{uuid.uuid4()}.txt", "w") as f:
        f.writelines([f"{pred[0]}, {pred[1]}\n" for pred in res])
    return [pred[1] for pred in res]
