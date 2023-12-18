import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.datasets import TestDataset, MaskBaseDataset

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model.model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, 'best.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mask_model = load_model('/data/ephemeral/home/n_boostcamp/level1-imageclassification-cv-10/results/rembg_exp/weights', 3, device).to(device)
    gender_model = load_model('/data/ephemeral/home/n_boostcamp/level1-imageclassification-cv-10/results/base_exp/weights', 2, device).to(device)
    age_model = load_model('/data/ephemeral/home/n_boostcamp/level1-imageclassification-cv-10/results/yolo_exp/weights', 3, device).to(device)
    mask_model.eval()
    gender_model.eval()
    age_model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            mask_pred = mask_model(images).argmax(dim=-1)
            gender_pred = gender_model(images).argmax(dim=-1)
            age_pred = age_model(images).argmax(dim=-1)
            for mask, gender, age in zip(mask_pred, gender_pred, age_pred):
                preds.append(MaskBaseDataset.encode_multi_class(mask.cpu().numpy(), gender.cpu().numpy(), age.cpu().numpy()))
            
    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'real_ensemble_final.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(384, 288), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='EfficientnetB4', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/data/ephemeral/home/train/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/data/ephemeral/home/project/results/train12/weights/'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
