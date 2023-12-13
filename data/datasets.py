import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

import pandas as pd
from sklearn.model_selection import train_test_split

# 지원되는 이미지 확장자 리스트
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    """파일 이름이 이미지 확장자를 가지는지 확인하는 함수

    Args:
        filename (str): 확인하고자 하는 파일 이름

    Returns:
        bool: 파일 이름이 이미지 확장자를 가지면 True, 그렇지 않으면 False.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()  
        self.calc_statistics() 

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    @staticmethod
    def get_profile_from_paths(paths:list[str]) -> tuple():
        """
        주어진 이미지 프로파일 path들로부터 id, gender, race, age를 각각 리스트로 뽑아줍니다
        """
        ids = []
        genders = []
        races = []
        ages = []

        for filename in paths:
            id, gender, race, age = filename.split("_")
            ids.append(id)
            genders.append(gender)
            races.append(race)
            ages.append(age)
        
        return (ids, genders, races, ages)

    def _split_balanced_profile(self, profiles:list[str], val_ratio:float) -> dict:
        """
        Scikit-learn의 train_test_split을 이용해 성별과 나이를 val_ratio만큼의 분포로 최대한 맞춰주는 함수입니다
        """
        profile_df = pd.DataFrame()
        profile_df["path"] = profiles

        ids, genders, races, ages = self.get_profile_from_paths(profile_df["path"])

        profile_df["gender"] = genders
        profile_df["age"] = ages

        new_label_list = []
        gender_list = []
        age_list = []

        for gender, age in zip(profile_df["gender"], profile_df["age"]):
            gender_list.append(GenderLabels.from_str(gender))
            age_list.append(AgeLabels.from_number(age))

        for i in range(len(profile_df)):
            new_label_list.append(MaskBaseDataset.encode_multi_class(0, gender_list[i], age_list[i]))

        profile_df["gender_age"] = new_label_list

        train_list, val_list = train_test_split(
            profile_df, test_size=val_ratio, random_state=42, stratify=profile_df["gender_age"]
        )

        train_indices = set(list(train_list.index))
        val_indices = set(list(val_list.index))

        return {"train": train_indices, "val": val_indices}

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]

        # 새로 만든 split_balanced_profile 함수를 이용해 gender, age에 맞춰 profile을 나눕니다
        split_profiles = self._split_balanced_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class OnlyAgeDataset(MaskSplitByProfileDataset):
    num_classes = 3
    class_name = ["young", "middle", "old"]
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
        
    def __getitem__(self, index):
        assert self.transform is not None

        image = self.read_image(index)
        age_label = self.get_age_label(index)
        image_transform = self.transform(image)
        return image_transform, age_label
    
class OnlyMaskDataset(MaskSplitByProfileDataset):
    num_classes = 3
    class_name = ["mask", "incorrect", "normal"]
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
        
    def __getitem__(self, index):
        assert self.transform is not None

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        image_transform = self.transform(image)
        return image_transform, mask_label
    
class OnlyGenderDataset(MaskSplitByProfileDataset):
    num_classes = 2
    class_name = ["male", "female"]
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
        
    def __getitem__(self, index):
        assert self.transform is not None

        image = self.read_image(index)
        gender_label = self.get_gender_label(index)
        image_transform = self.transform(image)
        return image_transform, gender_label
    
class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)