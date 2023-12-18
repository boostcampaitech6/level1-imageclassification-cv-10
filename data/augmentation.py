from torchvision import transforms
from torchvision.transforms import *
from PIL import Image
import torch
import torchvision

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
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