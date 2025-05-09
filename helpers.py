import torch
from ImageDataset import ImageDataset

def get_train_dataset():
    return ImageDataset("data\\train", "cpu")

def get_loader(dataset):
    sampler = torch.utils.data.RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=16,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
    )
    return loader

def get_valid_dataset(device):
    return ImageDataset("data/valid/", device)