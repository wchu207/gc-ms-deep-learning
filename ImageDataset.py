import os
import torch
import numpy as np
from pyteomics import mzml
from pyms import IntensityMatrix
from pyms.Spectrum import Scan
from pyms.GCMS.Class import GCMS_data
from torchvision.datasets import DatasetFolder

class ImageDataset(DatasetFolder):
    def __init__(self, data_dir, device):
        super().__init__(data_dir, return_path, extensions=("pt"))
        self.device = device
    

    def __getitem__(self, idx):
        path, label = super().__getitem__(idx)
        X = torch.load(path).to(self.device).float()
        return X.unsqueeze(0), label

def average_rt(X):
    pool = torch.nn.AvgPool1d(kernel_size=4, stride=4, ceil_mode=True)
    return pool(X.T).T

def normalize(X):
    return torch.nn.functional.normalize(X, p=np.inf, dim=())

def pad(X):
    current_rt_size = X.shape[0]
    new_rt_size = next_power_of_2(current_rt_size)
    
    current_mz_size = X.shape[1]
    new_mz_size = next_power_of_2(current_mz_size)

    return torch.nn.functional.pad(X, (0, new_mz_size - current_mz_size, 0, new_rt_size - current_rt_size))


def return_path(path):
    return path


def parse_subscans(scan):
    scan_obj = Scan(scan["m/z array"], scan["intensity array"])
    scan_times = []
    for subscan in scan["scanList"]["scan"]:
        scan_times.append(to_seconds(subscan["scan start time"]))

    return scan_times, [scan_obj] * len(scan_times)
    

def to_seconds(rt):
    if rt.unit_info == "minute":
        return rt*60
    elif rt.unit_info == "second":
        return r

def next_power_of_2(x):
        return 2**((x - 1).bit_length())