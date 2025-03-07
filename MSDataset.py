import os
import torch
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

        pad_time = (16 - (mzml.intensity_matrix.shape[0] % 16)) % 16
        pad_mz = (16 - (mzml.intensity_matrix.shape[1] % 16)) % 16
        
        mzml_data = torch.nn.functional.pad(
            torch.tensor(mzml.intensity_matrix).float().unsqueeze(0),
            (0, 0, 0, pad_time, 0, pad_mz),
            mode="constant")
        
        return mzml_data.to(self.device), label

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
            return IntensityMatrix.build_intensity_matrix(f)
        else:
            return None

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