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

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        
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
        self.age_drop = age_drop
        super().__init__(data_dir, mean, std, val_ratio)

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

class MaskSplitByProfileBalancedDataset(MaskSplitByProfileDataset):
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
    
    def _split_profile(self, profiles, val_ratio):
        id_list = []
        new_label_list = []
        for index, profile in enumerate(profiles):
            id, gender, race, age = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)
                
            id_list.append(index)
            new_label_list.append(self.encode_multi_class(0, gender_label, age_label))

        id_list = np.array(id_list)
        new_label_list = np.array(new_label_list)

        x_train, x_val, y_train, y_val = train_test_split(id_list, new_label_list, test_size = val_ratio, random_state =777, stratify = new_label_list)

        return {
            "train" : x_train, 
            "val" : x_val
        }
    
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


from torchvision import transforms
from torchvision.transforms import *

from PIL import Image
import torch

import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
import torch
import numpy as np
import random

import torchvision

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
class GenderAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class AgeAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, image):                
        return self.transform(image)

class MaskAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, image):                
        return self.transform(image)

class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

random_mirror = True

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def Hue(img, v):    # [-0.5, 0.5]
    assert -0.5 <= v <= 0.5
    return TF.adjust_hue(img, v)

def Grayscale(img, _):
    num_output_channels = TF.get_image_num_channels(img)
    return TF.rgb_to_grayscale(img, num_output_channels=num_output_channels)

def GaussianBlur(img, v):
    assert 0.1 <= v <= 2.0
    return img.filter(ImageFilter.GaussianBlur(radius=v))

def Identity(img, _):   # identity
    return img

def HorizontalFlip(img, _):
    return TF.hflip(img)

def VerticalFlip(img, _):
    return TF.vflip(img)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return ImageOps.solarize(img, threshold)

def augment_list():
    augmentation_list_to_explore = \
    [
        (ShearX, -0.3, 0.3),        # 0
        (ShearY, -0.3, 0.3),        # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),          # 4
        (AutoContrast, 0, 1),       # 5
        (Invert, 0, 1),             # 6
        (Equalize, 0, 1),           # 7
        (Solarize, 0, 256),         # 8
        (Posterize, 4, 8),          # 9
        (Contrast, 0.1, 1.9),       # 10
        (Color, 0.1, 1.9),          # 11    # same as Saturation
        (Brightness, 0.1, 1.9),     # 12
        (Sharpness, 0.1, 1.9),      # 13
        (Cutout, 0, 0.2),           # 14
        (Hue, -0.5, 0.5),           # 15
        (Grayscale, 0, 1),          # 16
        (GaussianBlur, 0.1, 2.0),   # 17
        (Identity, 0, 1),           # 18
        (HorizontalFlip, 0, 1),     # 19
        (VerticalFlip, 0, 1)        # 20
    ]
    return augmentation_list_to_explore

def get_augment(name):
    augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}
    return augment_dict[name]

def apply_augment(img, name, level):
    if level < 0.0:
        level = 0.0
    elif level > 1.0:
        level = 1.0
    augment_fn, low, high = get_augment(name)
    v = level * (high - low) + low
    return augment_fn(img.copy(), v)

class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, prob, level in policy:
                if random.random() > prob:
                    continue
                img = apply_augment(img, name, level)
        return img
    
class RandAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.randaug = torchvision.transforms.RandAugment(num_ops=3, magnitude=10, num_magnitude_bins=30, interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.randaug.forward(image)
        return self.basetransform(image)

class Augmix:
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.augmix = torchvision.transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, 
                                                    all_ops=True, interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.augmix.forward(image)
        return self.basetransform(image)

class AutoAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.autoaug = torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10,
                                                           interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.autoaug.forward(image)
        return self.basetransform(image)

class AugmentWide:
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.augwide = torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31,
                                                           interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.augwide.forward(image)
        return self.basetransform(image)

class SharpnessAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            RandomAdjustSharpness(sharpness_factor=8),
            Resize(resize),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
