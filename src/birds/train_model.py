# Copyright (c) 2017, PyTorch contributors
# Modifications copyright (C) Microsoft Corporation
# Licensed under the BSD license
# From https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import argparse
import copy
import os
import time

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import models
from torchvision import transforms


def load_data(data_dir):
    """Load the train/val data."""

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ["train", "val"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ["train", "val"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def train_model(model, criterion, optimizer, scheduler, num_epochs, data_dir):
    """Train the model."""

    # load training/validation data
    dataloaders, dataset_sizes, class_names = load_data(data_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # log the best val accuracy to AML run
            mlflow.log_metric("best_val_acc", float(best_acc))

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fine_tune_model(num_epochs, data_dir, learning_rate, momentum):
    """Load a pretrained model and reset the final fully connected layer."""

    # log the hyperparameter metrics to the AML run
    mlflow.log_metric("lr", float(learning_rate))
    mlflow.log_metric("momentum", float(momentum))

    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # only 2 classes to predict

    device = get_device()
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(),
                             lr=learning_rate, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model_ft, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs, data_dir)

    return model


def main():
    from datetime import datetime
    print(f"Starting training @{datetime.now()}")
    print(f"Torch version: {torch.__version__}")

    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="number of epochs to train")
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="input directory")
    parser.add_argument("--output-dir", type=str, default="outputs/",
                        help="output directory")
    parser.add_argument("--learning-rate", type=float,
                        default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    args, _ = parser.parse_known_args()

    print("Data directory is: " + args.data_dir)
    with mlflow.start_run() as _:
        model = fine_tune_model(args.num_epochs, args.data_dir,
                                args.learning_rate, args.momentum)
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.pt")
        # dummy_input = torch.randn(1, 3, 224, 224, device=get_device())
        # torch.onnx.export(model, dummy_input, model_path)
        torch.save(model, model_path)
        # mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
