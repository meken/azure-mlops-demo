import argparse
from glob import glob
from os import path

import torch
from torchvision import transforms

from birds.datasets import FilePathDataset


def run(model, dataset):
    result_list = []
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, num_workers=4, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for paths, contents in data_loader:
        inputs = contents.to(device, non_blocking=True)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            result_list += zip(map(path.basename, paths), preds.tolist())
    return result_list


def get_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()  # inference mode
    return model


def get_data_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_dataset_from_directory(data_dir, limit=10):
    data_transforms = get_data_transforms()
    files = [path.abspath(f) for f in glob(path.join(data_dir, "**/*.jpg"), recursive=True)][:limit]
    return FilePathDataset(files, transform=data_transforms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="input directory")
    parser.add_argument("--model-path", type=str, default="outputs/model.pt",
                        help="path to the model")
    args = parser.parse_args()
    model = get_model(args.model_path)
    dataset = get_dataset_from_directory(args.data_dir)
    results = run(model, dataset)
    print(f"Scored {len(results)} items")


if __name__ == "__main__":
    main()
