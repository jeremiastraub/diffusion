import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LSUNBase(Dataset):
    BASE_PATH = "~/data/lsun/"

    def __init__(self,
                 txt_train,
                 txt_val,
                 size=None,
                 #random_crop=False,
                 interpolation="bicubic",
                 data_root=None,
                 crop_size=None,
                 flip_p=0.5):
        self.split = self.get_split()
        self.base_path = os.path.expanduser(self.BASE_PATH)
        self.data_paths = {
            "train": os.path.join(self.base_path, txt_train),
            "validation": os.path.join(self.base_path, txt_val),
        }[self.split]
        # self.data_root = os.path.join(self.base_path, data_root)
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.base_path, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.crop_size = crop_size
        self.interpolation = {"linear":PIL.Image.LINEAR,
                              "bilinear":PIL.Image.BILINEAR,
                              "bicubic":PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        if self.crop_size is not None:
            raise ValueError("We use SSDE preprocessing")
            self.crop = transforms.RandomCrop(crop_size) if random_crop else transforms.CenterCrop(crop_size)
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
            image = image.resize((self.size, self.size), resample=self.interpolation)

        if self.crop_size is not None:
            image = self.crop(image)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        return example


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, txt_train="church_outdoor_train.txt", txt_val="church_outdoor_val.txt",
                 size=None, interpolation="bicubic", flip_p=0.5,
                 data_root="church_outdoor_train"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size,
                         interpolation=interpolation, flip_p=flip_p,
                         data_root=data_root)

    def get_split(self):
        return "train"


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, txt_train="church_outdoor_train.txt", txt_val="church_outdoor_val.txt",
                 size=None, interpolation="bicubic", flip_p=0.0,
                 data_root="church_outdoor_val"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size,
                         interpolation=interpolation, flip_p=flip_p,
                         data_root=data_root)

    def get_split(self):
        return "validation"


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, size=None, interpolation="bicubic", flip_p=0.5):
        super().__init__(txt_train="bedrooms_train.txt", txt_val=None, size=size,
                         interpolation=interpolation, flip_p=flip_p, data_root="bedrooms_train")
    def get_split(self):
        return "train"


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, size=None, interpolation="bicubic", flip_p=0.0):
        super().__init__(txt_train=None, txt_val="data/bedrooms_val.txt", size=size, flip_p=flip_p,
                         interpolation=interpolation, data_root="bedrooms_train")
    def get_split(self):
        return "validation"


class LSUNCatsTrain(LSUNBase):
    def __init__(self, size=None, interpolation="bicubic", flip_p=0.5):
        super().__init__(txt_train="data/cat_train.txt", txt_val=None, size=size,
                         interpolation=interpolation, flip_p=flip_p)

    def get_split(self):
        return "train"


class LSUNCatsValidation(LSUNBase):
    def __init__(self, size=None, interpolation="bicubic", flip_p=0.):
        super().__init__(txt_train=None, txt_val="data/cat_val.txt", size=size,
                         interpolation=interpolation, flip_p=flip_p)

    def get_split(self):
        return "validation"

## not yet ready below

class LSUNHorsesTrain(LSUNBase):
    def __init__(self, txt_train="data/horse_train.txt", txt_val="data/horse_test.txt",
                 size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "train"


class LSUNHorsesValidation(LSUNBase):
    def __init__(self, txt_train="data/horse_train.txt", txt_val="data/horse_test.txt",
                 size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "validation"


class LSUNDogsTrain(LSUNBase):
    def __init__(self, txt_train="data/dog_train.txt", txt_val="data/dog_test.txt",
                 size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "train"


class LSUNDogsValidation(LSUNBase):
    def __init__(self, txt_train="data/dog_train.txt", txt_val="data/dog_test.txt",
                 size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "validation"


class LSUNCarsTrain(LSUNBase):
    def __init__(self, txt_train="data/car_train.txt", txt_val="data/car_test.txt",
                 size=None, random_crop=True, interpolation="bicubic", only_crop_size=-1):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation, only_crop_size=only_crop_size)

    def get_split(self):
        return "train"


class LSUNCarsValidation(LSUNBase):
    def __init__(self, txt_train="data/car_train.txt", txt_val="data/car_test.txt",
                 size=None, random_crop=False, interpolation="bicubic", only_crop_size=-1):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation, only_crop_size=only_crop_size)

    def get_split(self):
        return "validation"


class LSUNBirdsTrain(LSUNBase):
    def __init__(self, txt_train="data/bird_train.txt", txt_val="data/bird_test.txt",
                 size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "train"


class LSUNBirdsValidation(LSUNBase):
    def __init__(self, txt_train="data/bird_train.txt", txt_val="data/bird_test.txt",
                 size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "validation"


class LSUNBoatsTrain(LSUNBase):
    def __init__(self, txt_train="data/boat_train.txt", txt_val="data/boat_test.txt",
                 size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "train"


class LSUNBoatsValidation(LSUNBase):
    def __init__(self, txt_train="data/boat_train.txt", txt_val="data/boat_test.txt",
                 size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "validation"


class LSUNAirplanesTrain(LSUNBase):
    def __init__(self, txt_train="data/airplane_train.txt", txt_val="data/airplane_test.txt",
                 size=None, random_crop=True, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "train"


class LSUNAirplanesValidation(LSUNBase):
    def __init__(self, txt_train="data/airplane_train.txt", txt_val="data/airplane_test.txt",
                 size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation)

    def get_split(self):
        return "validation"


class LSUNTowersTrain(LSUNBase):
    def __init__(self, txt_train="data/tower_train.txt", txt_val="data/tower_val.txt",
                 size=None, random_crop=False, interpolation="bicubic", only_crop_size=-1):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation, only_crop_size=only_crop_size)
    def get_split(self):
        return "train"


class LSUNTowersValidation(LSUNBase):
    def __init__(self, txt_train="data/tower_train.txt", txt_val="data/tower_val.txt",
                 size=None, random_crop=False, interpolation="bicubic", only_crop_size=-1):
        super().__init__(txt_train=txt_train, txt_val=txt_val, size=size, random_crop=random_crop,
                         interpolation=interpolation, only_crop_size=only_crop_size)
    def get_split(self):
        return "validation"


if __name__ == "__main__":
    dset1 = LSUNCatsTrain(crop_size=256)
    dset2 = LSUNCatsValidation(crop_size=256)
    for dset in [dset1, dset2]:
        print("length:", len(dset))
        ex = dset[0]
        print(ex.keys())
        print(ex["image"].shape)
    print("done.")
