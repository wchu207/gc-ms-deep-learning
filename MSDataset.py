import os
import torch
import numpy as np
from pyteomics import mzml
from pyms import IntensityMatrix
from pyms.Spectrum import Scan
from pyms.GCMS.Class import GCMS_data
from torchvision.datasets import DatasetFolder

class MSDataset(DatasetFolder):
    def __init__(self, data_dir, thread_pool, device):
        super().__init__(data_dir, return_path, extensions=("mzml"))
        self.device = device
        self.thread_pool = thread_pool

    def __getitem__(self, idx):
        path, label = super().__getitem__(idx)
        mzml = self.read_intensity_matrix(path)

        pad_time = (16 - mzml.intensity_matrix.shape[0] % 16) % 16
        pad_mz = max(512 - mzml.intensity_matrix.shape[1], 0)
        
        return path, torch.tensor(mzml.intensity_matrix)

    def get_intensity_matrix(self, idx):
        path, label = super().__getitem__(idx)
        return torch.tensor(self.read_intensity_matrix(path).intensity_matrix).to(self.device)

    def get_raw_object(self, idx):
        path, label = super().__getitem__(idx)
        return self.read_mzml(path, self.thread_pool)

    def read_mzml(self, path, pool):
        all_scans = []
        with mzml.read(path) as reader:
            all_scans = list(reader)
        
        parsed_scans = pool.map(parse_subscans, all_scans)
    
        times_lists, scans_lists = zip(*parsed_scans)
        times_list = [time for l in times_lists for time in l]
        scans_list = [scan for l in scans_lists for scan in l]
        
        if len(times_list) > 0:
            return GCMS_data(times_list, scans_list)
        else:
            return None
            
    def read_intensity_matrix(self, path):
        f = self.read_mzml(path, self.thread_pool)
        if f != None:
            return IntensityMatrix.build_intensity_matrix(f, min_mass=0)
        else:
            return None

def average_rt(X):
    pool = torch.nn.AvgPool1d(kernel_size=4, stride=4, ceil_mode=True)
    return pool(X.T).T

def normalize(X):
    return torch.nn.functional.normalize(X, p=np.inf, dim=())

def pad(X):
    current_rt_size = X.shape[0]
    new_rt_size = next_power_of_2(current_rt_size)
    
    current_mz_size = X.shape[1]
    new_mz_size = next_power_of_2(X.shape[1])

    return torch.nn.functional.pad(X, (0, new_rt_size - current_rt_size, 0, new_mz_size - current_mz_size))

    

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