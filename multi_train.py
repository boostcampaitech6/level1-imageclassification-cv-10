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
from data.augmentation import *

def train(data_dir, save_dir, args):
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
    dataset = dataset_module(data_dir=data_dir)
    
    num_classes = dataset.num_classes

    transform_module = getattr(import_module("data.augmentation"), args.augmentation)
    transform = transform_module(resize=args.resize, mean=dataset.mean, std=dataset.std)
    dataset.set_transform(transform)

    train_set, val_set = dataset.split_dataset()
    
    # train_set.set_transform(transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
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
            inputs, age_labels, mask_labels, gender_labels, labels = train_batch
            inputs = inputs.to(device)
            age_labels = age_labels.to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            age_output, mask_output, gender_output = model(inputs)
            
            age_loss = criterion(age_output, age_labels)
            mask_loss = criterion(mask_output, mask_labels)
            gender_loss= criterion(gender_output, gender_labels)
        
            loss = age_loss + mask_loss + gender_loss

            loss.backward()
            optimizer.step()
            
            age_output = torch.argmax(age_output, dim=-1)
            mask_output = torch.argmax(mask_output, dim=-1)
            gender_output = torch.argmax(gender_output, dim=-1)
            
            preds = age_output + gender_output*3 + mask_output*6
            
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
                inputs, age_labels, mask_labels, gender_labels, labels = val_batch
                inputs = inputs.to(device)
                age_labels = age_labels.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                labels = labels.to(device)
                
                age_output, mask_output, gender_output = model(inputs)
                
                age_loss = criterion(age_output, age_labels)
                mask_loss = criterion(mask_output, mask_labels)
                gender_loss= criterion(gender_output, gender_labels)
                
                loss = age_loss + mask_loss + gender_loss
                
                val_loss_items.append(loss.item())
                
                age_output = torch.argmax(age_output, dim=-1)
                mask_output = torch.argmax(mask_output, dim=-1)
                gender_output = torch.argmax(gender_output, dim=-1)
                
                preds = age_output + gender_output*3 + mask_output*6
                
                results.extend(list(preds.cpu().numpy()))
                targets.extend(list(labels.cpu().numpy()))
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            metrics = calculate_metrics(targets, results, num_classes)
            
            results.clear()
            targets.clear()
            val_loss_items.clear()
            
            if metrics["Total F1 Score"] > best_f1_score:
                torch.save(model.module.state_dict(), os.path.join(weight_path, 'best.pt'))
                best_f1_score = metrics["Total F1 Score"]
            
            validation_desc = \
                "Validation Loss: {:3.7f}, Validation Acc.: {:3.4f}, Precision: {:3.4f}, Recall: {:3.4f}, F1 Score: {:3.4f}, Best Validation F1 Score.:{:3.4f}".\
                format(val_loss, metrics["Total Accuracy"], metrics["Total Precision"], metrics["Total Recall"], metrics["Total F1 Score"], best_f1_score)
            
            print(validation_desc)
            txt_logger.update_string(validation_desc)
            
            torch.save(model.module.state_dict(), os.path.join(weight_path, 'last.pt'))
            wb_logger.log(
                {
                    "Train Loss": train_loss / len(train_loader),
                    "Train Accuracy": train_acc / len(train_set),
                    "Val Loss": val_loss,
                    "Val Accuracy": metrics["Total Accuracy"],
                    "Val Recall":metrics["Total Recall"],
                    "Val Precision": metrics["Total Precision"],
                    "Val F1_Score": metrics["Total F1 Score"],
                }
            )
    
    best_weight = torch.load(os.path.join(weight_path, 'best.pt'))
    model.module.load_state_dict(best_weight)
    with torch.no_grad():
        model.eval()
        results = []
        targets = []
        for val_batch in val_loader:
            inputs, age_labels, mask_labels, gender_labels, labels = val_batch
            inputs = inputs.to(device)
            age_labels = age_labels.to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            labels = labels.to(device)
        
            age_output, mask_output, gender_output = model(inputs)
            
            age_output = torch.argmax(age_output, dim=-1)
            mask_output = torch.argmax(mask_output, dim=-1)
            gender_output = torch.argmax(gender_output, dim=-1)
            
            preds = age_output + gender_output*3 + mask_output*6
            
            targets.extend(list(labels.cpu().numpy()))
            results.extend(list(preds.cpu().numpy()))
        
        print("Save Metric....")
        save_confusion_matrix(targets, results, num_classes, save_path)
        metrics = calculate_metrics(targets, results, num_classes)
        results.clear()
        targets.clear()
        
        parsed_metric = parse_metric(metrics, dataset.class_name)
        print(parsed_metric)
        
        txt_logger.update_string("Save Metric....")
        txt_logger.update_string(parsed_metric)
        
    txt_logger.close()
    
if __name__ == '__main__':
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
    train(data_dir=args.data_dir, save_dir=args.save_dir, args=args)