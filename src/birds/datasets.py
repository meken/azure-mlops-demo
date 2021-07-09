import torch
from torchvision import datasets


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        super().__init__()
        self.file_paths = file_paths
        self.transform = transform
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sample = self.loader(file_path)
        if self.transform:
            sample = self.transform(sample)
        return file_path, sample
