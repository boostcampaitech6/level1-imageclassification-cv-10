import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """
        Focal Loss를 계산합니다.

        Parameters
        ----------
        input_tensor : Tensor
            모델의 출력 텐서입니다.
        target_tensor : Tensor
            정답 레이블 텐서입니다.
        
        Returns
        -------
        Tensor
            계산된 손실값입니다.
        """
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
        """
        레이블 스무딩 손실을 계산합니다.

        Parameters
        ----------
        pred : Tensor
            모델의 출력 텐서입니다.
        target : Tensor
            정답 레이블 텐서입니다.
        
        Returns
        -------
        Tensor
            계산된 손실값입니다.
        """
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
        """
        F1 손실을 계산합니다.

        Parameters
        ----------
        y_pred : Tensor
            모델의 출력 텐서입니다.
        y_true : Tensor
            정답 레이블 텐서입니다.
        
        Returns
        -------
        Tensor
            계산된 F1 손실값입니다.
        """
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

        
_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "f1": F1Loss,
}

def criterion_entrypoint(criterion_name):
    """
    주어진 손실 함수 이름에 해당하는 손실 함수

    Args:
        criterion_name (str): 반환할 손실 함수 이름

    Returns:
        callable: 주어진 이름에 해당하는 손실 함수
    """
    return _criterion_entrypoints[criterion_name]

def is_criterion(criterion_name):
    """
    주어진 손실 함수 이름이 지원되는지 확인한다.

    Args:
        criterion_name (str): 확인할 손실 함수 이름

    Returns:
        bool: 지원되면 True, 그렇지 않으며 False
    """
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    """
    지정된 인수를 사용하여 손실 함수 객체를 생성한다.

    Args:
        criterion_name (str): 생성할 손실 함수 이름
        **kargs: 손실 함수 생성자에 전달된 키워드 인자

    Returns:
        nn.Module: 생성된 손실 함수 객체
    """
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion


