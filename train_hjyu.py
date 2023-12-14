from __future__ import print_function
import os
import yaml
import glob
import json
import multiprocessing
import random
import re
from importlib import import_module
from pathlib import Path

from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from data.datasets import MaskBaseDataset, MaskSplitByProfileDataset
from utils.loss import create_criterion
from utils.argparsers import Parser
from utils.WeightAndBiasLogger import WeightAndBiasLogger


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  
    plt.subplots_adjust(
        top=0.8
    ) 
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
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


def train(data_dir, save_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(save_dir, args.exp_name))
    exp_name = save_dir
    
    wb_logger = WeightAndBiasLogger(args, exp_name)
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module(".datasets", package="data"), args.dataset
    )
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module(".datasets", package="data"), args.augmentation
    ) 
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
        
    if args.sampler is None:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    elif args.sampler == "ImbalancedSampler":
        labels = [train_set[i][1] for i in range(len(train_set))]
        train_loader = DataLoader(
            train_set,
            sampler=ImbalancedDatasetSampler(train_set, labels = labels),
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            # shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    elif args.sampler == "WeightedSampler":
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
            # shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

    val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

    # -- model
    model_module = getattr(import_module(".model", package="model"), args.model) 
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion) 
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(args.lr),
        weight_decay=5e-4,
    )
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    best_epoch = 0
    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1_score = 0
    
    for epoch in tqdm(range(args.max_epochs)):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.max_epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )

                loss_value = 0
                matches = 0

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            
            val_labels = []
            val_preds = []
            
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )
                    
            # average option: micro / macro / weighted / None
            # micro : 전체 클래스에 대한 f1-score, 각 클래스 TP/FP/FN를 합한 뒤에 계산
            # macro : 각 클래스에 대한 f1-score를 계산한 뒤 평균
            # weighted : 각 클래스에 대한 f1-score 계산 후, 클래스 별 데이터 비율에 따른 가중치 평균
            # None : 각 클래스별 f1-score 
            val_f1_score = metrics.f1_score(
                y_true=val_labels, y_pred=val_preds, average="macro"
            )
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            
            if val_f1_score > best_val_f1_score:
                print(
                    f"New best model for val F1_score : {val_f1_score:.5%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pt")
                best_val_f1_score = val_f1_score
                best_epoch = epoch
            torch.save(model.module.state_dict(), f"{save_dir}/last.pt")
            print(
                f"[Val] Accracy : {val_acc:4.2%} ||"
                f"[Val] F1-Score : {val_f1_score:.5%}, loss: {val_loss:4.2} || "
                f"best F1-Score : {best_val_f1_score:.5%}, best loss: {best_val_loss:4.2}, best epoch: {best_epoch}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/F1-Score", val_f1_score, epoch)
            logger.add_figure("results", figure, epoch)
            
            scheduler.step(val_loss)
            
            wb_logger.log(
                {
                    "Train loss": train_loss,
                    "Train accuracy": train_acc,
                    "Val loss": val_loss,
                    "Val accuracy": val_acc,
                    "Val F1_Score": val_f1_score,
                }
            )
            
            print()


if __name__ == '__main__':
    p = Parser()
    p.create_parser()
    
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
    
    train(data_dir=args.data_dir, save_dir=args.save_dir, args=args)
    