import argparse
import json
import multiprocessing
import os
from importlib import import_module
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch

from utils.plot import save_confusion_matrix
from utils.util import *
from utils.loss import create_criterion
from utils.metric import calculate_metrics, parse_metric
from utils.logger import Logger, WeightAndBiasLogger
from utils.argparsers import Parser
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import WeightedRandomSampler

import random
import time
from torchvision.transforms import v2



def train(train_data_dir, val_data_dir, save_dir, args):
    seed_everything(args.seed)
    save_path = increment_path(os.path.join(save_dir, args.exp_name))
    create_directory(save_path)
    weight_path = os.path.join(save_path, 'weights')
    create_directory(weight_path)
    args.save_path = save_path
    wb_logger = WeightAndBiasLogger(args, save_path.split("/")[-1])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_module = getattr(import_module("data.datasets"), args.dataset)

    #train/val 폴더가 나뉘었으므로 데이터셋도 train_set, val_set 두 개로 나누어 생성합니다
    train_dataset = dataset_module(data_dir=train_data_dir)
    val_dataset = dataset_module(data_dir=val_data_dir)
    
    num_classes = train_dataset.num_classes

    train_transform_module = getattr(import_module("data.augmentation"), args.augmentation)
    val_transform_module = getattr(import_module("data.augmentation"), "BaseAugmentation")
    train_transform = train_transform_module(resize=args.resize, mean=train_dataset.mean, std=train_dataset.std)
    val_transform = val_transform_module(resize=args.resize, mean=val_dataset.mean, std=val_dataset.std)
    train_dataset.set_transform(train_transform)
    val_dataset.set_transform(val_transform)
    
    train_set, val_set = train_dataset, val_dataset

    collate = None

    if args.sampler is None:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            collate_fn=collate,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    #torchsampler에서 제공하는 ImbalancedDatasetSampler를 사용합니다
    elif args.sampler == "ImbalancedSampler":
        labels = [train_set[i][1] for i in range(len(train_set))]
        train_loader = DataLoader(
            train_set,
            sampler=ImbalancedDatasetSampler(train_set, labels = labels),
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            collate_fn=collate,
            # shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    #torch.utils에서 제공하는 WeightedSampler를 사용합니다
    elif args.sampler == "WeightedSampler":
        #train set의 라벨 분포를 이용해 0부터 17까지 18개의 라벨에 대해 계산한 가중치 값들
        BASE_WEIGHT = [6.885245901639344,
                       9.21951219512195,
                       45.54216867469879,
                       5.163934426229508,
                       4.626682986536108,
                       34.678899082568805,
                       34.42622950819672,
                       46.09756097560975,
                       227.710843373494,
                       25.81967213114754,
                       23.133414932680537,
                       173.39449541284404,
                       34.42622950819672,
                       46.09756097560975,
                       227.710843373494,
                       25.81967213114754,
                       23.133414932680537,
                       173.39449541284404]
        weights = [BASE_WEIGHT[train_set[i][1]] for i in range(len(train_set))]
        weightedsampler = WeightedRandomSampler(weights=weights, num_samples=len(train_set), replacement=True)
        train_loader = DataLoader(
            train_set,
            sampler=weightedsampler,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            collate_fn=collate,
            # shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2 ,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    model_module = getattr(import_module("model.model"), args.model)
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    criterion = create_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)

    if "Adam" in args.optimizer:
        optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), lr=float(args.lr), weight_decay=5e-4, amsgrad=True)
    elif "RMSprop" == args.optimizer:
        optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), lr=float(args.lr), weight_decay=5e-4,alpha=0.9, momentum=0.9, eps=1e-08, centered=False)
    else:
        optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), lr=float(args.lr), weight_decay=5e-4, amsgrad=True)
        
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    txt_logger = Logger(save_path)
    txt_logger.update_string(str(args))
    
    best_val_loss = np.inf
    best_f1_score = 0.
    best_val_f1_score = 0
    
    for epoch in range(args.max_epochs):
        model.train()

        train_desc_format = "Epoch[{:03d}/{:03d}] - Train Loss: {:3.7f}, Train Acc.: {:3.4f}"
        train_process_bar = tqdm(train_loader, desc=train_desc_format.format(epoch, args.max_epochs, 0., 0.), mininterval=0.01)
        train_loss = 0.
        train_acc = 0.

        for train_batch in train_process_bar:
            inputs, labels = train_batch
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
            
            print(sum([results[i] == targets[i] for i in range(len(results))]))

            val_loss = np.sum(val_loss_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            metrics = calculate_metrics(targets, results, num_classes)
            
            if metrics["Total F1 Score"] > best_f1_score:
                torch.save(model.module.state_dict(), os.path.join(weight_path, 'best.pt'))
                best_f1_score = metrics["Total F1 Score"]
            
            validation_desc = \
                "Validation Loss: {:3.7f}, Validation Acc.: {:3.4f}, Precision: {:3.4f}, Recall: {:3.4f}, F1 Score: {:3.4f}, Best Validation F1 Score.:{:3.4f}".\
                format(val_loss, metrics["Total Accuracy"], metrics["Total Precision"], metrics["Total Recall"], metrics["Total F1 Score"], best_f1_score)
            
            print(validation_desc)
            txt_logger.update_string(validation_desc)
            
            torch.save(model.module.state_dict(), os.path.join(weight_path, 'last.pt'))

            # 모델이 잘못 예측한 이미지의 인덱스들 중 10개를 랜덤하게 뽑아서 wandb를 이용해 로깅합니다.
            false_pred_images = []
            random_sample = list(random.sample(metrics["False Image Indexes"], 10))
            for index in random_sample:
                false_pred_images.append(wb_logger.update_image_with_label(val_set[index][0], results[index].item(), targets[index].item()))

            wb_logger.log(
                {
                    "Train Loss": train_loss / len(train_loader),
                    "Train Accuracy": train_acc / len(train_set),
                    "Val Loss": val_loss,
                    "Val Accuracy": metrics["Total Accuracy"],
                    "Val Recall":metrics["Total Recall"],
                    "Val Precision": metrics["Total Precision"],
                    "Val F1_Score": metrics["Total F1 Score"],
                    "Image": false_pred_images
                }
            )

            results.clear()
            targets.clear()
            val_loss_items.clear()
            false_pred_images.clear()
    
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
        
        parsed_metric = parse_metric(metrics, val_dataset.class_name)
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