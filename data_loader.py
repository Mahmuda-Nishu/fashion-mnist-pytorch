"""Custom loader for the raw Fashion-MNIST files (no prebuilt dataset loader)."""

import gzip

import numpy as np
import torch


def read_images(filename):
    with gzip.open(filename, "rb") as f:
        f.read(16)  # skip the file header
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
        return torch.tensor(data.reshape(-1, 28 * 28) / 255.0)  # normalize to 0-1


def read_labels(filename):
    with gzip.open(filename, "rb") as f:
        f.read(8)  # skip the file header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)


def get_fashion_mnist_data(path="data"):
    train_x = read_images(f"{path}/train-images-idx3-ubyte.gz")
    train_y = read_labels(f"{path}/train-labels-idx1-ubyte.gz")
    test_x = read_images(f"{path}/t10k-images-idx3-ubyte.gz")
    test_y = read_labels(f"{path}/t10k-labels-idx1-ubyte.gz")
    return train_x, train_y, test_x, test_y
