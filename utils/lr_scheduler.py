import math
from torch.optim import lr_scheduler

def get_lr(optimizer):
    """
    주어진 옵티마이저에서 학습률을 추출한다.

    Args:
        optimizer (Optimizer): PyTorch 옵티마이저 객체.

    Returns:
        float: 옵티마이저의 현재 학습률.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class StepLr(lr_scheduler.StepLR):
    """
    StepLR 스케줄러의 사용자 정의 버전. 이 스케줄러는 매 epoch마다 학습률을 조정한다.

    Args:
        optimizer (Optimizer): PyTorch 옵티마이저 객체.
        epoch (int): 총 epoch 수.
    """
    def __init__(self, optimizer, epoch):
        lr = get_lr(optimizer)
        gamma = math.pow(lr * 0.1, 1 / epoch)
        super().__init__(optimizer, step_size=1, gamma=gamma)

class CosAnnealling(lr_scheduler.CosineAnnealingWarmRestarts):
    """
    CosineAnnealingWarmRestarts 스케줄러의 사용자 정의 버전. 이 스케줄러는 코사인 주기에 따라 학습률을 조정한다.

    Args:
        optimizer (Optimizer): PyTorch 옵티마이저 객체.
        epoch (int): 총 epoch 수.
    """
    def __init__(self, optimizer, epoch):
        lr = get_lr(optimizer)
        super().__init__(optimizer, T_0=epoch, eta_min=lr/10)
        
class LinearLr(lr_scheduler.LinearLR):
    """
    LinearLR 스케줄러의 사용자 정의 버전. 이 스케줄러는 선형적으로 학습률을 감소시킨다.

    Args:
        optimizer (Optimizer): PyTorch 옵티마이저 객체.
        epoch (int): 총 epoch 수.
    """
    def __init__(self, optimizer, epoch):
        lr = get_lr(optimizer)
        super().__init__(optimizer, start_factor=1, end_factor=0.1, total_iters=epoch)
        
_scheduler_entrypoints = {
    'step': StepLr,
    'cos': CosAnnealling,
    'lin': LinearLr,
}

def scheduler_entrypoint(scheduler_name):
    """
    주어진 스케줄러 이름에 해당하는 스케줄러 생성 함수를 반환한다.

    Args:
        scheduler_name (str): 스케줄러의 이름.

    Returns:
        function: 해당 스케줄러를 생성하는 함수.
    """
    return _scheduler_entrypoints[scheduler_name]

def is_scheduler(scheduler_name):
    """
    주어진 스케줄러 이름이 지원되는 스케줄러인지 확인한다.

    Args:
        scheduler_name (str): 확인할 스케줄러의 이름.

    Returns:
        bool: 스케줄러가 지원되면 True, 그렇지 않으면 False.
    """
    return scheduler_name in _scheduler_entrypoints

def create_scheduler(scheduler_name, **kwargs):
    """
    주어진 이름과 인자를 바탕으로 스케줄러 객체를 생성한다.

    Args:
        scheduler_name (str): 생성할 스케줄러의 이름.
        **kwargs: 스케줄러 생성에 필요한 추가 키워드 인자.

    Returns:
        lr_scheduler: 생성된 스케줄러 객체.

    Raises:
        RuntimeError: 알려지지 않은 스케줄러 이름이 주어진 경우 예외를 발생시킨다.
    """
    if is_scheduler(scheduler_name):
        create_fn = scheduler_entrypoint(scheduler_name)
        scheduler = create_fn(**kwargs)
    elif scheduler_name == None:
        scheduler = LinearLr(kwargs['optimizer'], math.inf)
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)
    return scheduler