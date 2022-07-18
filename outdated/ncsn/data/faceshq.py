import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from convae.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class FacesBase(Dataset):
    BASE_PATH = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQTrain(FacesBase):
    BASE_PATH = "~/data/celebahq/"

    def __init__(self, size, keys=None, random_flip=True):
        super().__init__()
        self.base_path = os.path.expanduser(self.BASE_PATH)
        data_dir = os.path.join(self.base_path, "celebahq")
        with open(os.path.join(self.base_path, "celebahqtrain.txt"), "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(data_dir, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False,
                               random_flip=random_flip)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    BASE_PATH = "~/data/celebahq/"

    def __init__(self, size, keys=None):
        super().__init__()
        self.base_path = os.path.expanduser(self.BASE_PATH)
        data_dir = os.path.join(self.base_path, "celebahq")
        with open(os.path.join(self.base_path, "celebahqvalidation.txt"), "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(data_dir, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQTrain(FacesBase):
    BASE_PATH = "~/data/ffhq/"

    def __init__(self, size, keys=None, random_flip=True):
        super().__init__()
        self.base_path = os.path.expanduser(self.BASE_PATH)
        data_dir = os.path.join(self.base_path, "ffhq")
        with open(os.path.join(self.base_path, "ffhqtrain.txt"), "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(data_dir, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False,
                               random_flip=random_flip)
        self.keys = keys


class FFHQValidation(FacesBase):
    BASE_PATH = "~/data/ffhq/"

    def __init__(self, size, keys=None):
        super().__init__()
        self.base_path = os.path.expanduser(self.BASE_PATH)
        data_dir = os.path.join(self.base_path, "ffhq")
        with open(os.path.join(self.base_path, "ffhqvalidation.txt"), "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(data_dir, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex
