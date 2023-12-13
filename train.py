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

def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    save_path = increment_path(os.path.join(model_dir, args.name))
    create_directory(save_path)
    weight_path = os.path.join(save_path, 'weights')
    create_directory(weight_path)
    args.save_path = save_path
    wb_logger = WeightAndBiasLogger(args, save_path.split("/")[-1])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_module = getattr(import_module("data.dataset"), args.dataset)
    dataset = dataset_module(data_dir=data_dir)
    
    num_classes = dataset.num_classes

    transform_module = getattr(import_module("data.dataset"), args.augmentation)
    transform = transform_module(resize=args.resize, mean=dataset.mean, std=dataset.std)
    dataset.set_transform(transform)

    train_set, val_set = dataset.split_dataset()

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

    model_module = getattr(import_module("models.model"), args.model)
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    criterion = create_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)

    optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4, amsgrad=True)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    txt_logger = Logger(save_path)
    
    best_val_acc = 0.
    best_val_loss = np.inf
    
    for epoch in range(args.epochs):
        model.train()

        train_desc_format = "Epoch[{:03d}/{:03d}] - Train Loss: {:3.4f}, Train Acc.: {:3.4f}"
        train_process_bar = tqdm(train_loader, desc=train_desc_format.format(epoch, args.epochs, 0., 0.), mininterval=0.01)
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
            
            train_desc = train_desc_format.format(epoch, args.epochs, loss.item(),\
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
            val_acc_items = []
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
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(targets)
            best_val_loss = min(best_val_loss, val_loss)
            metrics = calculate_metrics(targets, results, num_classes)
            
            results.clear()
            targets.clear()
            val_loss_items.clear()
            val_acc_items.clear()
            
            if val_acc > best_val_acc:
                torch.save(model.module.state_dict(), os.path.join(weight_path, 'best.pt'))
                best_val_acc = val_acc
            
            validation_desc = \
                "Validation Loss: {:3.4f}, Validation Acc.: {:3.4f}, Best Validation Acc.:{:3.4f}, Precision: {:3.4f}, Recall: {:3.4f}, F1 Score: {:3.4f}".\
                format(val_loss, val_acc, best_val_acc, metrics["Total Precision"], metrics["Total Recall"], metrics["Total F1 Score"])
            
            print(validation_desc)
            txt_logger.update_string(validation_desc)
            
            torch.save(model.module.state_dict(), os.path.join(weight_path, 'last.pt'))
            wb_logger.log(
                {
                    "Train Loss": train_loss / len(train_loader),
                    "Train Accuracy": train_acc / len(train_set),
                    "Val Loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Val Recall":metrics["Total Recall"],
                    "Val Precision": metrics["Total Precision"],
                    "Val F1_Score": metrics["Total F1 Score"],
                }
            )
    
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
        metrics = calculate_metrics(targets, results, num_classes)
        results.clear()
        targets.clear()
        
        parsed_metric = parse_metric(metrics, dataset.class_name)
        print(parsed_metric)
        
        txt_logger.update_string("Save Metric....")
        txt_logger.update_string(parsed_metric)
        
    txt_logger.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    import os
    
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[256, 192], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
    parser.add_argument('--model', type=str, default='EfficientNet', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/usr/src/app/BoostCampl_Lv1/train/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/usr/src/app/BoostCampl_Lv1/project/results/'))
    parser.add_argument('--name', default='train', help='model save at {SM_MODEL_DIR}/{name}')

    args = parser.parse_args()
    
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)