import torch
import torchvision
from torchvision import transforms
from PIL import Image

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        """
        기본적인 이미지 전처리를 수행하는 클래스입니다.

        이 클래스는 이미지 크기 조정, 텐서 변환, 정규화 등의 기본적인 전처리 단계를 포함합니다.
        사용자는 `resize`, `mean`, `std` 매개변수를 통해 전처리를 커스터마이즈할 수 있습니다.

        Parameters
        ----------
        resize : tuple
            이미지 크기 조정 시 사용할 새로운 크기입니다.
        mean : tuple
            정규화에 사용될 평균값입니다.
        std : tuple
            정규화에 사용될 표준편차입니다.
        """
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
      
class GenderAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class AgeAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, image):                
        return self.transform(image)

class MaskAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def __call__(self, image):                
        return self.transform(image)

class AddGaussianNoise(object):
    """
    가우시안 노이즈를 이미지에 추가하는 클래스입니다.

    이 변환은 입력 이미지에 가우시안(정규) 노이즈를 추가하여, 모델의 강인성을 향상시키는 데 사용됩니다.

    Parameters
    ----------
    mean : float
        노이즈의 평균값입니다.
    std : float
        노이즈의 표준편차입니다.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    """
    사용자 정의 이미지 전처리를 수행하는 클래스입니다.

    이 클래스는 `BaseAugmentation`에 추가하여 사용자 정의 전처리를 제공합니다.
    이는 이미지 크기 조정, 텐서 변환, 정규화, 그리고 가우시안 노이즈 추가를 포함합니다.

    Parameters
    ----------
    resize : tuple
        이미지 크기 조정 시 사용할 새로운 크기입니다.
    mean : tuple
        정규화에 사용될 평균값입니다.
    std : tuple
        정규화에 사용될 표준편차입니다.
    """
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.CenterCrop((384, 288)),
            transforms.Resize(resize, Image.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            # AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class RandAugmentation:
    """
    무작위 데이터 증강을 수행하는 클래스입니다.

    `RandAugment` 변환을 사용하여 이미지에 다양한 무작위 증강을 적용합니다.
    이는 기본 전처리 단계 이후에 적용됩니다.

    Parameters
    ----------
    resize : tuple
        이미지 크기 조정 시 사용할 새로운 크기입니다.
    mean : tuple
        정규화에 사용될 평균값입니다.
    std : tuple
        정규화에 사용될 표준편차입니다.
    """
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.randaug = torchvision.transforms.RandAugment(num_ops=3, magnitude=10, num_magnitude_bins=30, interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.randaug.forward(image)
        return self.basetransform(image)

class Augmix:
    """
    AugMix 데이터 증강을 수행하는 클래스입니다.

    `AugMix` 변환을 사용하여 이미지에 다양한 조합의 증강을 적용합니다.
    이는 기본 전처리 단계 이후에 적용됩니다.

    Parameters
    ----------
    resize : tuple
        이미지 크기 조정 시 사용할 새로운 크기입니다.
    mean : tuple
        정규화에 사용될 평균값입니다.
    std : tuple
        정규화에 사용될 표준편차입니다.
    """
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.augmix = torchvision.transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, 
                                                    all_ops=True, interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.augmix.forward(image)
        return self.basetransform(image)

class AutoAugmentation:
    """
    자동 데이터 증강을 수행하는 클래스입니다.

    `AutoAugment` 변환을 사용하여, 정해진 정책에 따라 이미지에 자동 증강을 적용합니다.
    이는 기본 전처리 단계 이후에 적용됩니다.

    Parameters
    ----------
    resize : tuple
        이미지 크기 조정 시 사용할 새로운 크기입니다.
    mean : tuple
        정규화에 사용될 평균값입니다.
    std : tuple
        정규화에 사용될 표준편차입니다.
    """
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.autoaug = torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10,
                                                           interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.autoaug.forward(image)
        return self.basetransform(image)

class AugmentWide:
    """
    광범위한 데이터 증강을 수행하는 클래스입니다.

    `TrivialAugmentWide` 변환을 사용하여, 다양한 간단한 증강을 이미지에 적용합니다.
    이는 기본 전처리 단계 이후에 적용됩니다.

    Parameters
    ----------
    resize : tuple
        이미지 크기 조정 시 사용할 새로운 크기입니다.
    mean : tuple
        정규화에 사용될 평균값입니다.
    std : tuple
        정규화에 사용될 표준편차입니다.
    """
    def __init__(self, resize, mean, std, **args):
        self.basetransform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.augwide = torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31,
                                                           interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = self.augwide.forward(image)
        return self.basetransform(image)