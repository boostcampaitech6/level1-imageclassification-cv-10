# import argparse
import os
import yaml
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.datasets import TestDataset, MaskBaseDataset
from utils.argparsers import Parser

def load_model(saved_model, num_classes, device):
    """
    저장된 모델을 불러오고 초기화한다.

    Args:
        saved_model (str): 모델이 저장된 경로.
        num_classes (int): 분류할 클래스의 수.
        device (torch.device): 모델을 로드할 디바이스.

    Returns:
        torch.nn.Module: 초기화된 모델.
    """
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
    데이터셋에 대해 모델의 추론을 수행하고 결과를 저장한다.

    Args:
        data_dir (str): 추론할 데이터셋이 위치한 디렉토리 경로.
        model_dir (str): 학습된 모델이 저장된 디렉토리 경로.
        output_dir (str): 추론 결과를 저장할 디렉토리 경로.
        args: 추론 설정 매개변수가 포함된 객체.

    Returns:
        None: 함수는 추론 결과를 파일로 저장하며 별도의 반환 값은 없다.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

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
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


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
    
    p.print_args(args)

    data_dir = args.test_data_dir
    exp_name = args.exp_name + str(args.test_exp_num if args.test_exp_num else "")
    model_dir = args.save_dir + "/{}/weights".format(exp_name)
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
