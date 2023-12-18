import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
import math

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

from importlib import import_module

from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

from torchvision import transforms
from torchvision.transforms import *
from PIL import Image
import cv2

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
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


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

def change_incorrect_normal(mask):
    if mask == 'normal':
        return MaskLabels.INCORRECT
    elif mask == 'incorrect_mask':
        return MaskLabels.NORMAL
    else:
        return MaskLabels.MASK


def change_incorrect_to_mask(mask):
    if mask == 'incorrect_mask':
        return MaskLabels.MASK
    elif mask == 'normal':
        return MaskLabels.NORMAL
    else:
        return MaskLabels.MASK
    
def change_gender(gender):
    if gender == 'male':
        return 'female'
    else:
        return 'male'
    
class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3
    class_name = [
        "mask_male_young", "mask_male_middle", "mask_male_old",
        "mask_female_young", "mask_female_middle", "mask_female_old",
        "incorrect_male_young", "incorrect_male_middle", "incorrect_male_old",
        "incorrect_female_young", "incorrect_female_middle", "incorrect_female_old",
        "normal_male_young", "normal_male_middle", "normal_male_old",
        "normal_female_young", "normal_female_middle", "normal_female_old",
    ]
    
    relabel_dict = {
        'incorrect_to_from_normal': ['000020', '004418', '005227'],             
        'incorrect_to_mask': ['000645', '003574'],                              
        'incorrect_gender': ['001200', '004432', '005223', '001498-1', '000725',
                             '006359', '006360', '006361', '006362', '006363', '006364', '001720'] 
    }
    
    ignores = ["003399"]
    
    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
        "mask1_rembg": MaskLabels.MASK,
        "mask2_rembg": MaskLabels.MASK,
        "mask3_rembg": MaskLabels.MASK,
        "mask4_rembg": MaskLabels.MASK,
        "mask5_rembg": MaskLabels.MASK,
        "incorrect_mask_rembg": MaskLabels.INCORRECT,
        "normal_rembg": MaskLabels.NORMAL
    }

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, age_drop=False):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.age_drop = age_drop
        
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        
        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        # 각 이미지를 리스트에 담는 코드
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue
            
            # ./input/train/images
            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:
                    continue
                    
                img_path = os.path.join(self.data_dir, profile, file_name)
                    
                mask_label = self._file_names[_file_name]

                id, gender, _, age = profile.split("_")
                    
                if id in self.ignores:
                    continue
                
                if id in self.relabel_dict['incorrect_to_from_normal']:
                    mask_label = change_incorrect_normal(_file_name)
                    
                if id in self.relabel_dict['incorrect_to_mask']:
                    mask_label = change_incorrect_to_mask(_file_name)
                    
                if id in self.relabel_dict['incorrect_gender']:
                    gender = change_gender(gender)
                    
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

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
        img_array = np.array(Image.open(image_path))
        if img_array.shape[-1] == 3:
            return Image.fromarray(img_array)
        else:
            return Image.fromarray(img_array[:, :, 0:3])

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
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
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, age_drop=False):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)
        self.age_drop = age_drop

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue
                    
                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    
                    if self.age_drop and (57 <= int(age) <= 59) or (28 <= int(age) <= 30): 
                        continue
                    
                    if id in self.ignores:
                        continue

                    if id in self.relabel_dict['incorrect_to_from_normal']:
                        mask_label = change_incorrect_normal(_file_name)
                    
                    if id in self.relabel_dict['incorrect_to_mask']:
                        mask_label = change_incorrect_to_mask(_file_name)
                
                    if id in self.relabel_dict['incorrect_gender']:
                        gender = change_gender(gender)
                        
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

class MultiLabelMaskSplitByProfileDataset(MaskSplitByProfileDataset):
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
    
    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, age_label, mask_label, gender_label, multi_class_label
		
class OnlyAgeDataset(MaskSplitByProfileDataset):
    num_classes = 3
    class_name = ["young", "middle", "old"]
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, age_drop=False):
        super().__init__(data_dir, mean, std, val_ratio, age_drop=age_drop)
        
    def __getitem__(self, index):
        assert self.transform is not None

        image = self.read_image(index)
        age_label = self.get_age_label(index)
        image_transform = self.transform(image)
        return image_transform, age_label

class OnlyAgeDatasetForRegression(MaskSplitByProfileDataset):
    num_classes = 3
    class_name = ["young", "middle", "old"]
    # thresholds = [Age.HIGH, Age.MID, Age.LOW]
    age_values = []
    
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, age_drop=False):
        super().__init__(data_dir, mean, std, val_ratio, age_drop=age_drop)
        
    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, _ = os.path.splitext(file_name)
                    if _file_name not in self._file_names:
                        continue
                    img_path = os.path.join(self.data_dir, profile, file_name)
                    _, _, _, age = profile.split("_")
                    if self.age_drop and (57 <= int(age) <= 59) or (28 <= int(age) <= 30): 
                        continue
                    self.age_values.append(float(age))
                    self.age_labels.append(AgeLabels.from_number(age))
                    self.image_paths.append(img_path)
                    self.indices[phase].append(cnt)
                    cnt += 1
    
    def get_age_value(self, index):
        return self.age_values[index]
    
    def __getitem__(self, index):
        assert self.transform is not None

        image = self.read_image(index)
        age_label = self.get_age_label(index)
        age_value = self.get_age_value(index)
        image_transform = self.transform(image)
        return image_transform, (age_value, age_label)
    
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
