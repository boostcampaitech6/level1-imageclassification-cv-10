import multiprocessing
import numpy as np

import optuna

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import argparse

from tqdm import tqdm
from importlib import import_module
from util import seed_everything, AverageMeter
from metric import calculate_metrics
from loss import FocalLoss, F1Loss, LabelSmoothingLoss
from ..model.model import EfficientnetB4
from ..data.datasets import MaskSplitByProfileDataset

def train(model, criterion, optimizer, scheduler):
    model = model.to(device)
    criterion = criterion.to(device)

    train_desc_format = "Epoch[{:03d}/{:03d}] - Train Loss: {:3.7f}, Train Acc.: {:3.4f}"
    train_process_bar = tqdm(train_loader, desc=train_desc_format.format(epoch, args.num_epochs, 0., 0.), mininterval=0.01)
    train_loss = 0.
    train_acc = 0.
    for epoch in range(args.num_epochs):
        model.train()
        for inputs, labels in train_process_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_desc = train_desc_format.format(epoch, args.num_epochs, loss.item(),\
                (preds == labels).sum().item() / args.batch_size)
            train_process_bar.set_description(train_desc)
            
            train_loss += loss.item()
            train_acc += (preds == labels).sum().item()

        metrics = validation(model, criterion)

        if scheduler is not None:
            scheduler.step()

    return metrics["Total Accuracy"]

def validation(model, criterion):
    best_val_loss = 0.
    best_f1_score = 0.
    with torch.no_grad():
        model.eval()
        val_loss_items = []
        results = []
        targets = []

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)
            
            results.extend(list(preds.cpu().numpy()))
            targets.extend(list(labels.cpu().numpy()))
            
            loss_item = criterion(outputs, labels).item()
            val_loss_items.append(loss_item)
        
        val_loss = np.sum(val_loss_items) / len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)
        metrics = calculate_metrics(targets, results, num_classes)

        validation_desc = \
            "Validation Loss: {:3.7f}, Validation Acc.: {:3.4f}, Precision: {:3.4f}, Recall: {:3.4f}, F1 Score: {:3.4f}".\
            format(val_loss, metrics["Total Accuracy"], metrics["Total Precision"], metrics["Total Recall"], metrics["Total F1 Score"])
        print(validation_desc)

    return metrics

def search_hyperparam(trial):
    lr = trial.suggest_categorical("lr", [0.1, 0.5, 0.01, 0.05, 0.001, 0.005])
    epochs = trial.suggest_int("epochs", low=50, high=100, step=25)
    criterion = trial.suggest_categorical("criterion", ["ce", "smooth", "focal", "f1"])
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"])
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "None"])

    return {
        "lr": lr,
        "epochs" : epochs,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

def objective(trial):
    hyperparams = search_hyperparam(trial)

    # model_module = getattr(import_module("..model.model", package='utils'), args.model)
    model = EfficientnetB4(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    if hyperparams["loss_fn"] == "ce": 
        criterion = nn.CrossEntropyLoss()
    elif hyperparams["loss_fn"] == "smooth":
        criterion = LabelSmoothingLoss(classes=num_classes, device=device)
    elif hyperparams["loss_fn"] == "focal":
        criterion = FocalLoss()
    elif hyperparams["loss_fn"] == "f1":
        criterion = F1Loss(classes=num_classes)

    if hyperparams["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9, weight_decay=5e-4)
    elif hyperparams["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=5e-4)
    elif hyperparams["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=5e-4)
    
    if hyperparams["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif hyperparams["scheduler"] == "None":
        scheduler = None

    metrics = train(trial, criterion, optimizer, scheduler)

    return metrics[""]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Select batch size")
    parser.add_argument("--seed", type=int, default=137, help="Select Random Seed")
    parser.add_argument("--trial", type=int, default=1000, help="Select number of trial")
    parser.add_argument('--dataset', default="MaskBaseDataset", help="The input dataset type")
    parser.add_argument('--data_dir', default="../input/train/images", help="The dataset folder path")
    parser.add_argument("--num_epochs", type=int, default=30, help="Select number of epochs")
    args = parser.parse_args()

    seed_everything(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset_module = getattr(import_module("..data.datasets"), args.dataset)
    dataset = MaskSplitByProfileDataset(data_dir=args.data_dir)
    num_classes = dataset.num_classes

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
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trial)

    print("Best trial")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))