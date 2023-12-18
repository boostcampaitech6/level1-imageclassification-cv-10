import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data.datasets import MaskBaseDataset  # 가정: 이 모듈이 존재하고 MaskBaseDataset 클래스에 decode_multi_class 메소드가 있음

sys.path.append(".")
sys.path.append("..")

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    """
    이미지, 실제 레이블, 예측 레이블을 사용하여 이미지 그리드를 생성한다.

    Args:
        np_images (np.ndarray): 이미지 배열.
        gts (list[int]): 실제 레이블 리스트.
        preds (list[int]): 예측 레이블 리스트.
        n (int, optional): 표시할 이미지 수. 기본값은 16.
        shuffle (bool, optional): 이미지를 무작위로 섞을지 여부. 기본값은 False.

    Returns:
        matplotlib.figure.Figure: 생성된 이미지 그리드.
    """
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2)) 
    plt.subplots_adjust(top=0.8)              
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(int(n_grid), int(n_grid), int(idx + 1), title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def save_confusion_matrix(target: np.ndarray, prediction: np.ndarray, label_num: int, save_path: str):
    """
    주어진 대상 및 예측 레이블을 기반으로 혼동 행렬을 생성하고 저장한다.

    Args:
        target (np.ndarray): 실제 레이블 배열.
        prediction (np.ndarray): 예측 레이블 배열.
        label_num (int): 레이블의 개수.
        save_path (str): 혼동 행렬을 저장할 경로.

    Returns:
        None
    """
    matrix = confusion_matrix(target, prediction)
    plt.figure(figsize=(12, 9)) 
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.jpg'))