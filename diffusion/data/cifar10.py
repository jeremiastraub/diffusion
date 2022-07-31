import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class CIFAR10Train(Dataset):
    def __init__(self, flip_p=0.5):
        super().__init__()
        dset = torchvision.datasets.CIFAR10(
            "data/cifar10",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=flip_p),
                    transforms.ToTensor()
                ]
            ),
        )
        self.data = dset

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0 * x - 1.
        x = x.permute(1, 2, 0)
        return {"image": x, "class": y}

    def __len__(self):
        return len(self.data)


class CIFAR10Validation(Dataset):
    def __init__(self):
        super().__init__()
        dset = torchvision.datasets.CIFAR10(
            "data/cifar10",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.data = dset

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0 * x - 1.
        x = x.permute(1, 2, 0)
        return {"image": x, "class": y}

    def __len__(self):
        return len(self.data)


class CIFARnTrain(Dataset):
    def __init__(self, n, flip_p=0.5):
        super().__init__()
        assert n <= 10, "Only 10 classes available in CIFAR10"
        dset = torchvision.datasets.CIFAR10(
            "data/cifar10",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=flip_p),
                    transforms.ToTensor()
                ]
            ),
        )
        class_mask = np.isin(dset.targets, np.arange(n))
        dset.targets = np.array(dset.targets)[class_mask]
        dset.data = dset.data[class_mask]
        self.data = dset

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0 * x - 1.
        x = x.permute(1, 2, 0)
        return {"image": x, "class": y}

    def __len__(self):
        return len(self.data)


class CIFARnValidation(Dataset):
    def __init__(self, n):
        super().__init__()
        assert n <= 10, "Only 10 classes available in CIFAR10"
        dset = torchvision.datasets.CIFAR10(
            "data/cifar10",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        class_mask = np.isin(dset.targets, np.arange(n))
        dset.targets = np.array(dset.targets)[class_mask]
        dset.data = dset.data[class_mask]
        self.data = dset

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0 * x - 1.
        x = x.permute(1, 2, 0)
        return {"image": x, "class": y}

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    pass
