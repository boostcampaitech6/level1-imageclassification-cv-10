import math
from torch.optim import lr_scheduler

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class StepLr(lr_scheduler.StepLR):
    def __init__(self, optimizer, epoch):
        lr = get_lr(optimizer)
        gamma = math.pow(lr * 0.1, 1 / epoch)
        super().__init__(optimizer, step_size=1, gamma=gamma)

class CosAnnealling(lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, epoch):
        lr = get_lr(optimizer)
        super().__init__(optimizer, T_0=epoch, eta_min=lr/10)
        
class LinearLr(lr_scheduler.LinearLR):
    def __init__(self, optimizer, epoch):
        lr = get_lr(optimizer)
        super().__init__(optimizer, start_factor=1, end_factor=0.1, total_iters=epoch)
        
_scheduler_entrypoints = {
    'step': StepLr,
    'cos': CosAnnealling,
    'lin': LinearLr,
}

def scheduler_entrypoint(scheduler_name):
    return _scheduler_entrypoints[scheduler_name]

def is_scheduler(scheduler_name):
    return scheduler_name in _scheduler_entrypoints

def create_scheduler(scheduler_name, **kwargs):
    if is_scheduler(scheduler_name):
        create_fn = scheduler_entrypoint(scheduler_name)
        scheduler = create_fn(**kwargs)
    elif scheduler_name == None:
        scheduler = LinearLr(kwargs['optimizer'], math.inf)
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)
    return scheduler