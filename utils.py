import torch
import torch.utils
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def download_and_divide_data(dataset, n_clients):
    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor(), download=True
        )

    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transforms.ToTensor(), download=True
        )
    return (
        torch.utils.data.random_split(train_dataset, [1 / n_clients] * n_clients),
        test_dataset,
    )
