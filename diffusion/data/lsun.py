import os

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from diffusion.data.base import ConcatDatasetWithIndex

# ----------------------------------------------------------------------------

BASE_PATH_LSUN = "data/lsun"


class LSUNBase(Dataset):
    BASE_PATH = BASE_PATH_LSUN

    def __init__(
        self,
        txt_file,
        size=None,
        interpolation="bicubic",
        data_root=None,
        flip_p=0.5,
    ):
        """LSUN Dataset base class.

        Args:
            txt_file: Path to .txt file containing the filenames relative to
                ```BASE_PATH``.
            size: If given, resize images to (size x size)
            interpolation: interpolation type
            base_path: base path for LSUN data
            data_root: path to data relative to ``BASE_PATH``
            flip_p: Horizontal flip probability
        """
        self.base_path = os.path.expanduser(self.BASE_PATH)
        self.data_paths = os.path.join(self.base_path, txt_file)
        self.data_root = os.path.join(self.base_path, data_root)
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize(
                (self.size, self.size), resample=self.interpolation
            )
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        return example


class LSUNChurchesTrain(LSUNBase):
    def __init__(
        self,
        txt_file="church_outdoor_train.txt",
        data_root="church_outdoor",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNChurchesValidation(LSUNBase):
    def __init__(
        self,
        txt_file="church_outdoor_val.txt",
        data_root="church_outdoor",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNBedroomsTrain(LSUNBase):
    def __init__(
        self,
        txt_file="bedrooms_train.txt",
        data_root="bedrooms_train",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNBedroomsValidation(LSUNBase):
    def __init__(
        self,
        txt_file="bedrooms_val.txt",
        data_root="bedrooms_validation",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNCatsTrain(LSUNBase):
    def __init__(
        self,
        txt_file="cat_train.txt",
        data_root="",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNCatsValidation(LSUNBase):
    def __init__(
        self,
        txt_file="cat_val.txt",
        data_root="",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNHorsesTrain(LSUNBase):
    def __init__(
        self,
        txt_file="horse_train.txt",
        data_root="horse",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNHorsesValidation(LSUNBase):
    def __init__(
        self,
        txt_file="horse_val.txt",
        data_root="horse",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNDogsTrain(LSUNBase):
    def __init__(
        self,
        txt_file="dog_train_cleaned.txt",
        data_root="dog",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNDogsValidation(LSUNBase):
    def __init__(
        self,
        txt_file="dog_val_cleaned.txt",
        data_root="dog",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNCarsTrain(LSUNBase):
    def __init__(
        self,
        txt_file="car_train.txt",
        data_root="car",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNCarsValidation(LSUNBase):
    def __init__(
        self,
        txt_file="car_val.txt",
        data_root="car",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNBirdsTrain(LSUNBase):
    def __init__(
        self,
        txt_file="bird_train.txt",
        data_root="bird",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNBirdsValidation(LSUNBase):
    def __init__(
        self,
        txt_file="bird_val.txt",
        data_root="bird",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNBoatsTrain(LSUNBase):
    def __init__(
        self,
        txt_file="boat_train.txt",
        data_root="boat",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNBoatsValidation(LSUNBase):
    def __init__(
        self,
        txt_file="boat_val.txt",
        data_root="boat",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNAirplanesTrain(LSUNBase):
    def __init__(
        self,
        txt_file="airplane_train.txt",
        data_root="airplane",
        flip_p=0.5,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNAirplanesValidation(LSUNBase):
    def __init__(
        self,
        txt_file="airplane_val.txt",
        data_root="airplane",
        flip_p=0.0,
        **kwargs
    ):
        super().__init__(
            txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs
        )


class LSUNAnimalTrain(Dataset):
    """This LSUN version contains only animal classes:
        - horse
        - bird
        - dog
        - cat
    """
    def __init__(self, **kwargs):
        super().__init__()
        print("Constructing LSUNAnimal Train...")
        d1 = LSUNHorsesTrain(**kwargs)
        d2 = LSUNBirdsTrain(**kwargs)
        d3 = LSUNDogsTrain(**kwargs)
        d4 = LSUNCatsTrain(**kwargs)
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class LSUNAnimalValidation(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        print("Constructing LSUNAnimal Validation...")
        d1 = LSUNHorsesValidation(**kwargs)
        d2 = LSUNBirdsValidation(**kwargs)
        d3 = LSUNDogsValidation(**kwargs)
        d4 = LSUNCatsValidation(**kwargs)
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class LSUNAnimalBalancedTrain(Dataset):
    """Class-balanced version of LSUNanimal"""
    def __init__(self, subset_size=None, **kwargs):
        super().__init__()
        print("Constructing class-balanced LSUNAnimal Train...")
        d1 = LSUNHorsesTrain(**kwargs)
        d2 = LSUNBirdsTrain(**kwargs)
        d3 = LSUNDogsTrain(**kwargs)
        d4 = LSUNCatsTrain(**kwargs)

        dset_size = len(d4) # cats is the smallest
        if subset_size is not None:
            assert subset_size < dset_size, f"Cats only of length {dset_size}"
            dset_size = subset_size

            d4 = Subset(
                d4,
                np.random.choice(
                    np.arange(len(d4)), replace=False, size=dset_size
                )
            )

        d1 = Subset(
            d1,
            np.random.choice(np.arange(len(d1)), replace=False, size=dset_size)
        )
        d2 = Subset(
            d2,
            np.random.choice(np.arange(len(d2)), replace=False, size=dset_size)
        )
        d3 = Subset(
            d3,
            np.random.choice(np.arange(len(d3)), replace=False, size=dset_size)
        )

        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class LSUN7Train(Dataset):
    """Class overlap with Cifar-10, i.e. this LSUN version contains
        - airplane
        - automobile
        - bird
        - cat
        - dog
        - horse
        - ship/boat
    """
    def __init__(self, **kwargs):
        super().__init__()
        print("Constructing LSUN7 Train...")
        d1 = LSUNHorsesTrain(**kwargs)
        d2 = LSUNBirdsTrain(**kwargs)
        d3 = LSUNCarsTrain(**kwargs)
        d4 = LSUNAirplanesTrain(**kwargs)
        d5 = LSUNBoatsTrain(**kwargs)
        d6 = LSUNDogsTrain(**kwargs)
        d7 = LSUNCatsTrain(**kwargs)
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4, d5, d6, d7])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class LSUN7Validation(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        print("Constructing LSUN7 Train...")
        d1 = LSUNHorsesValidation(**kwargs)
        d2 = LSUNBirdsValidation(**kwargs)
        d3 = LSUNCarsValidation(**kwargs)
        d4 = LSUNAirplanesValidation(**kwargs)
        d5 = LSUNBoatsValidation(**kwargs)
        d6 = LSUNDogsValidation(**kwargs)
        d7 = LSUNCatsValidation(**kwargs)
        self.data = ConcatDatasetWithIndex([d1, d2, d3, d4, d5, d6, d7])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex
