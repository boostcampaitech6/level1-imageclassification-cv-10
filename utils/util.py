from pathlib import Path
import glob
import re
import numpy as np
import torch
import random
import os

def increment_path(path, exist_ok=False):
    """
    주어진 경로에 해당하는 새로운 경로를 생성한다. 동일한 경로가 이미 존재하는 경우, 숫자를 증가시켜 새 경로를 생성한다.

    Args:
        path (str): 생성하려는 파일 또는 디렉토리의 기본 경로.
        exist_ok (bool, optional): 동일한 경로가 존재해도 괜찮은 경우 True. 기본값은 False.

    Returns:
        str: 생성된 새로운 경로.
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
    
def seed_everything(seed):
    """
    난수 생성을 위한 시드를 고정한다.

    Args:
        seed (int): 사용할 시드 값.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    """
    옵티마이저의 현재 학습률을 반환한다.

    Args:
        optimizer (torch.optim.Optimizer): 조회할 옵티마이저.

    Returns:
        float: 현재 학습률.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def create_directory(dir_path):
    """
    주어진 경로에 디렉토리를 생성한다. 경로가 이미 존재하는 경우 아무것도 하지 않는다.

    Args:
        dir_path (str): 생성할 디렉토리의 경로.

    Returns:
        None
    """

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)