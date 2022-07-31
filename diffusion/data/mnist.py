import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTTrain(Dataset):
    def __init__(self):
        super().__init__()
        dset = torchvision.datasets.MNIST(
            "data/mnist/",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.data = dset

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0 * x - 1.
        # Duplicate grey-scale channel to have 3 channels
        # x = x.repeat(3, 1, 1)
        x = x.permute(1, 2, 0)
        return {"image": x, "class": y}

    def __len__(self):
        return len(self.data)


class MNISTValidation(Dataset):
    def __init__(self):
        super().__init__()
        dset = torchvision.datasets.MNIST(
            "data/mnist/",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.data = dset

    def __getitem__(self, i):
        x, y = self.data[i]
        x = 2.0 * x - 1.
        # Duplicate grey-scale channel to have 3 channels
        # x = x.repeat(3, 1, 1)
        x = x.permute(1, 2, 0)
        return {"image": x, "class": y}

    def __len__(self):
        return len(self.data)
