import torch
import torchvision
from torchvision import transforms
from PIL import Image

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
class GenderAugmentation:
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
            transforms.CenterCrop((320, 256)),
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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
