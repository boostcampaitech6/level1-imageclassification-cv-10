import json
import os

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from importlib import import_module
from tqdm import tqdm

from data.dataloader import create_data_loader

from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import v2

from utils.plot import save_confusion_matrix
from utils.util import *
from utils.loss import create_criterion
from utils.lr_scheduler import create_scheduler
from utils.metric import calculate_metrics, parse_metric
from utils.logger import Logger, WeightAndBiasLogger
from utils.argparsers import Parser

from data.augmentation import BaseAugmentation

import random
import time
from torchvision.transforms import v2
from torch.utils.data import default_collate

from torchvision.transforms import v2
from ultralytics import YOLO
from rembg import remove as rembg_model

def setup_paths(save_dir, exp_name):
    save_path = increment_path(Path(save_dir) / exp_name)
    create_directory(save_path)
    weight_path = save_path / 'weights'
    create_directory(weight_path)
    return save_path, weight_path

def create_optimizer(optimizer_name, model_parameters, lr, weight_decay, extra_params=None):
    """
    지정된 이름과 매개변수를 사용하여 옵티마이저를 생성한다.

    Args:
        optimizer_name (str): 생성할 옵티마이저의 이름 (예: 'Adam', 'RMSprop', 'AdamW', 'sgd').
        model_parameters (iterable): 옵티마이저에 전달할 모델 파라미터.
        lr (float): 학습률.
        weight_decay (float): 가중치 감소(정규화) 매개변수.
        extra_params (dict, optional): 옵티마이저에 추가로 전달할 매개변수.

    Returns:
        torch.optim.Optimizer: 생성된 옵티마이저.
    """
    params = filter(lambda p: p.requires_grad, model_parameters)
    if optimizer_name == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, alpha=0.9, momentum=0.9, eps=1e-08, centered=False)
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
def create_scheduler(scheduler_name, optimizer, max_epochs):
    """
    지정된 이름과 매개변수를 사용하여 학습률 스케줄러를 생성한다.

    Args:
        scheduler_name (str): 생성할 스케줄러의 이름 (예: 'cosine', 'step', 'exponential').
        optimizer (torch.optim.Optimizer): 스케줄러에 연결할 옵티마이저.
        max_epochs (int): 최대 에폭 수.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: 생성된 스케줄러.
    """
    if scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == "step":
        return lr_scheduler.StepLR(optimizer, step_size=2)
    elif scheduler_name == "exponential":
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def train(train_data_dir, val_data_dir, save_dir, args):
    """
    Train a model for image classification.

    이 함수는 이미지 분류를 위한 모델을 학습합니다. 데이터셋을 로드하고, 모델을 초기화하며, 학습 과정을 실행하고, 
    결과를 로깅하고, 최적의 모델을 저장합니다. 학습 과정에서는 진행 상태가 표시되며, 각 에폭마다 학습 및 검증 손실과
    정확도가 계산됩니다. 또한, 모델이 잘못 예측한 이미지를 선택하여 로깅할 수 있습니다.

    Parameters
    ----------
    train_data_dir : str
        학습 데이터셋이 위치한 디렉토리의 경로입니다. 이 경로에는 학습에 사용될 이미지 파일들이 포함되어 있습니다.

    val_data_dir : str
        검증 데이터셋이 위치한 디렉토리의 경로입니다. 모델의 성능을 평가하기 위한 이미지 파일들이 이 경로에 포함되어 있습니다.

    save_dir : str
        학습된 모델과 로그 파일을 저장할 디렉토리의 경로입니다. 이 경로 내에 모델 가중치와 학습 진행 상황에 대한 로그 파일이 저장됩니다.

    args : Namespace
        학습 설정을 포함하는 매개변수입니다. 이 객체는 학습률, 배치 크기, 최대 에폭 수, 모델 이름, 최적화 알고리즘 선택 등과 같은
        다양한 학습 매개변수를 포함할 수 있습니다. 이 매개변수는 명령줄 인수나 설정 파일을 통해 전달될 수 있습니다.
    """
    # Initializing
    seed_everything(args.seed)
    save_path, weight_path = setup_paths(save_dir, args.exp_name)
    wb_logger = WeightAndBiasLogger(args, save_path.split("/")[-1], args.project_name)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get dataset
    dataset_module = getattr(import_module("data.datasets"), args.dataset)
    
    if args.age_drop:
        train_dataset = dataset_module(data_dir=train_data_dir, age_drop=bool(args.age_drop))
    else:
        train_dataset = dataset_module(data_dir=train_data_dir)

    val_dataset = dataset_module(data_dir=val_data_dir)
    num_classes = train_dataset.num_classes

    # Get transform module
    train_transform_module = getattr(import_module("data.augmentation"), args.augmentation)
    val_transform_module = getattr(import_module("data.augmentation"), "BaseAugmentation")

    train_transform = train_transform_module(resize=args.resize, mean=train_dataset.mean, std=train_dataset.std)
    val_transform = val_transform_module(resize=args.resize, mean=val_dataset.mean, std=val_dataset.std)

    train_dataset.set_transform(train_transform)
    val_dataset.set_transform(val_transform)

    collate = None
    if args.cutmix:
        if args.cutmix == "cutmix":
            collate_base = v2.CutMix(num_classes=train_dataset.num_classes)
        elif args.cutmix == "mixup":
            collate_base = v2.MixUp(num_classes=val_dataset.num_classes)
        else:
            raise ValueError("Please provide cutmix or mixup as argument")
        
        collate = lambda batch : collate_base(*default_collate(batch))

    # Get DataLoader
    train_loader = create_data_loader(train_dataset, args.batch_size, use_cuda, sampler=args.sampler, collate=collate, is_train=True)
    val_loader = create_data_loader(val_dataset, args.valid_batch_size, use_cuda, is_train=False)

    # Get Model
    model_module = getattr(import_module("model.model"), args.model)
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # Set criterion, optimizer and scheduler
    criterion = create_criterion(args.criterion)
    optimizer = create_optimizer(args.optimizer, model.parameters(), float(args.lr), 5e-4)
    scheduler = create_scheduler(args.scheduler, optimizer, args.max_epochs, step_size=2, gamma=0.5)

    # Save config file and log
    with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    txt_logger = Logger(save_path)
    txt_logger.update_string(str(args))
    
    # Train & Validation
    best_val_loss = np.inf
    best_f1_score = 0.

    for epoch in range(args.max_epochs):
        model.train()

        train_desc_format = "Epoch[{:03d}/{:03d}] - Train Loss: {:3.7f}, Train Acc.: {:3.4f}"
        train_process_bar = tqdm(train_loader, desc=train_desc_format.format(epoch, args.max_epochs, 0., 0.), mininterval=0.01)
        train_loss = 0.
        train_acc = 0.

        for train_batch in train_process_bar:
            inputs, labels = train_batch
            if args.cutmix:
                labels = torch.argmax(labels, dim=-1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()
            
            train_desc = train_desc_format.format(epoch, args.max_epochs, loss.item(),\
                (preds == labels).sum().item() / args.batch_size)
            train_process_bar.set_description(train_desc)
            
            train_loss += loss.item()
            train_acc += (preds == labels).sum().item()
        
        train_process_bar.close()
        txt_logger.update_string(train_desc)
        scheduler.step()

        with torch.no_grad():
            model.eval()
            val_loss_items = []
            results = []
            targets = []
                    
            print("Calculate validation set.....")
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                
                results.extend(list(preds.cpu().numpy()))
                targets.extend(list(labels.cpu().numpy()))
                
                loss_item = criterion(outs, labels).item()
                val_loss_items.append(loss_item)
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            metrics = calculate_metrics(targets, results, num_classes)
            
            results.clear()
            targets.clear()
            val_loss_items.clear()
            
            if metrics["Total F1 Score"] > best_f1_score or (metrics["Total F1 Score"] == best_f1_score and best_val_loss < val_loss):
                torch.save(model.module.state_dict(), os.path.join(weight_path, 'best.pt'))
                best_f1_score = metrics["Total F1 Score"]
            
            validation_desc = \
                "Validation Loss: {:3.7f}, Validation Acc.: {:3.4f}, Precision: {:3.4f}, Recall: {:3.4f}, F1 Score: {:3.4f}, Best Validation F1 Score.:{:3.4f}".\
                format(val_loss, metrics["Total Accuracy"], metrics["Total Precision"], metrics["Total Recall"], metrics["Total F1 Score"], best_f1_score)
            
            print(validation_desc)
            txt_logger.update_string(validation_desc)
            
            torch.save(model.module.state_dict(), os.path.join(weight_path, 'last.pt'))

            false_pred_images = []
            random_sample = list(random.sample(metrics["False Image Indexes"], 10))
            for index in random_sample:
                false_pred_images.append(wb_logger.update_image_with_label(val_dataset[index][0], results[index].item(), targets[index].item()))

            wb_logger.log(
                {
                    "Train Loss": train_loss / len(train_loader),
                    "Train Accuracy": train_acc / len(train_dataset),
                    "Val Loss": val_loss,
                    "Val Accuracy": metrics["Total Accuracy"],
                    "Val Recall":metrics["Total Recall"],
                    "Val Precision": metrics["Total Precision"],
                    "Val F1_Score": metrics["Total F1 Score"],
                    # "Image": false_pred_images
                }
            )
            results.clear()
            targets.clear()
            val_loss_items.clear()
            # false_pred_images.clear()
    
    best_weight = torch.load(os.path.join(weight_path, 'best.pt'))
    model.module.load_state_dict(best_weight)
    with torch.no_grad():
        model.eval()
        results = []
        targets = []
        for val_batch in val_loader:
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            targets.extend(list(labels.cpu().numpy()))
            results.extend(list(preds.cpu().numpy()))
        
        print("Save Metric....")
        save_confusion_matrix(targets, results, num_classes, save_path)
        wb_logger.log_confusion_matrix(targets, results)
        metrics = calculate_metrics(targets, results, num_classes)
        results.clear()
        targets.clear()
        
        parsed_metric = parse_metric(metrics, val_set.class_name)
        print(parsed_metric)
        
        txt_logger.update_string("Save Metric....")
        txt_logger.update_string(parsed_metric)
        
    txt_logger.close()
    
if __name__ == '__main__':
    start_time = time.time()
    p = Parser()
    p.create_parser()
    
    import yaml
    pargs = p.parser.parse_args()
    try:
        with open(pargs.config, 'r') as fp:
            load_args = yaml.load(fp, Loader=yaml.FullLoader)
        key = vars(pargs).keys()
        for k in load_args.keys():
            if k not in key:
                print("Wrong argument: ", k)
                assert(k in key)
            p.parser.set_defaults(**load_args)
    except FileNotFoundError:
        print("Invalid filename. Check your file path or name.")
        
    args = p.parser.parse_args() 
    p.print_args(args)
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(train_data_dir=args.train_data_dir, val_data_dir=args.val_data_dir, save_dir=args.save_dir, args=args)
    print("--- %s seconds ---" % (time.time() - start_time))