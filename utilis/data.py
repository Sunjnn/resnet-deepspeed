import os
import torch
import torchvision
from torchvision import transforms


def dataloader(data_path, batch, train=True, shuffle=True):
    datasets = dataset(data_path, train)
    data_loader = torch.utils.data.DataLoader(datasets, batch_size=batch, shuffle=shuffle)
    return data_loader


def dataset(data_path, train=True):
    input_image_size = 224
    scale = 256 / 224

    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        subdir = 'train'
    else:
        transform = transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        subdir = 'val'

    path = os.path.join(data_path, subdir)
    datasets = torchvision.datasets.ImageFolder(path, transform=transform)

    return datasets
