import os
from torchvision.datasets import DatasetFolder

class MSDataset(DatasetFolder) :
    def __init__(self, data_dir, loader, transform=None, target_transform=None):
        super().__init__(data_dir, loader, transform=transform, target_transform=target_transform, extensions=("mzml"))
