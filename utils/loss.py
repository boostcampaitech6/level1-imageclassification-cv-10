import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss는 클래스 불균형 문제를 해결하기 위해 고안된 손실 함수입니다.

    Parameters
    ----------
    weight : Tensor, optional
        각 클래스에 대한 가중치입니다.
    gamma : float, default=2.0
        감마 값으로, 잘못 예측된 샘플에 대한 패널티를 조정합니다.
    reduction : str, default="mean"
        손실을 줄이는 방법입니다. 'mean', 'sum', 또는 'none'이 될 수 있습니다.
    """
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1-prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
        
class LabelSmoothingLoss(nn.Module):
    """
    레이블 스무딩을 사용한 손실 함수입니다.

    Parameters
    ----------
    classes : int, default=3
        분류할 클래스의 수입니다.
    smoothing : float, default=0.1
        스무딩 값으로, 레이블의 분산을 조정합니다.
    dim : int, default=-1
        연산을 수행할 차원입니다.
    """
    def __init__(self, classes=3, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

class F1Loss(nn.Module):
    """
    F1 점수를 기반으로 한 손실 함수입니다.

    Parameters
    ----------
    classes : int, default=3
        분류할 클래스의 수입니다.
    epsilon : float, default=1e-7
        수치 안정성을 위한 작은 값입니다.
    """
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2
        
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

class MSELoss(nn.Module):
    """
    Mean Squared Error (MSE) 손실을 계산하는 클래스.

    PyTorch의 nn.Module을 상속받아 MSE 손실을 구현한다. 이 클래스는 예측값과 목표값 사이의 평균 제곱 오차를 계산한다.

    Attributes:
        func (nn.MSELoss): 내부적으로 사용되는 PyTorch의 MSELoss 객체.

    Methods:
        forward: 예측값과 목표값을 받아 MSE 손실을 계산한다.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.func = nn.MSELoss(reduction=reduction)
    
    def forward(self, predictions, targets):
        return self.func(predictions, targets)


def focal(logits, labels, alpha, gamma):
    """
    Focal Loss를 계산합니다.

    Parameters
    ----------
    logits : Tensor
        모델의 출력 텐서입니다.
    labels : Tensor
        정답 레이블 텐서입니다.
    alpha : float
        focal loss의 알파값입니다.
    gamma : float
        focal loss의 감마값입니다.
    
    Returns
    -------
    Tensor
        계산된 Focal 손실값입니다.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits))).cuda()

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss).cuda()

    focal_loss /= torch.sum(labels).cuda()
    return focal_loss

class BalancedFocalLoss(nn.Module):
    """
    클래스 불균형 문제를 해결하기 위해 2019년에 소개된 손실 함수인 BalancedFocalLoss입니다.

    매개변수
    ----------
    gamma : float, 기본값=2.0
        잘못 예측된 샘플에 대한 패널티를 조절하는 gamma 값.

    beta : float, 기본값=0.9
        클래스당 효과적인 샘플 수를 계산하는 데 사용되는 beta 값.

    samples_per_class : list, 기본값=[2745, 2050, 415, 3660, 4085, 545, 549, 410, 83,
                                      732, 817, 109, 549, 410, 83, 732, 817, 109]
        각 클래스의 샘플 수를 포함하는 리스트. 클래스 가중치 계산에 사용됩니다.

    """
    def __init__(self, gamma=2.0, beta=0.9, samples_per_class=[2745, 2050, 415, 3660, 4085, 545, 549, 410, 83, 
                                                                                              732, 817, 109, 549, 410, 83, 732, 817, 109]):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.beta = beta
        self.samples = samples_per_class

    def forward(self, logits, labels):
        beta = self.beta
        gamma = self.gamma
        samples_per_cls = self.samples
        ######
        no_of_classes = 18
        ######
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        labels_one_hot = F.one_hot(labels, no_of_classes).float().cuda()

        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes).cuda()

        cb_loss = focal(logits, labels_one_hot, weights, gamma)
        
        return cb_loss
    
_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "f1": F1Loss,
    "mse": MSELoss,
    "bfocal": BalancedFocalLoss
}

def criterion_entrypoint(criterion_name):
    """
    주어진 손실 함수 이름에 해당하는 손실 함수 생성 함수를 반환한다.

    Args:
        criterion_name (str): 손실 함수의 이름.

    Returns:
        function: 해당 손실 함수를 생성하는 함수.
    """
    return _criterion_entrypoints[criterion_name]

def is_criterion(criterion_name):
    """
    주어진 손실 함수 이름이 지원되는 손실 함수인지 확인한다.

    Args:
        criterion_name (str): 확인할 손실 함수의 이름.

    Returns:
        bool: 손실 함수가 지원되면 True, 그렇지 않으면 False.
    """
    return criterion_name in _criterion_entrypoints

def create_criterion(criterion_name, **kwargs):
    """
    주어진 이름과 인자를 바탕으로 손실 함수 객체를 생성한다.

    Args:
        criterion_name (str): 생성할 손실 함수의 이름.
        **kwargs: 손실 함수 생성에 필요한 추가 키워드 인자.

    Returns:
        nn.Module: 생성된 손실 함수 객체.
    """
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion
