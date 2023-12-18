import argparse
import os
import yaml

class Parser(object):    
    def __init__(self, description=""):
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)
    
    def create_parser(self):
        # Directory
        self.parser.add_argument(
            '--train-data-dir',
            default="/input",
            help="The dataset folder path"
        )
        self.parser.add_argument(
            '--val-data-dir',
            default="/input",
            help="The dataset folder path"
        )
        self.parser.add_argument(
            '--test-data-dir',
            default="/input",
            help="The test dataset folder path"
        )
        self.parser.add_argument(
            '--save-dir',
            default="./results",
            help="The folder is for saving results"
        )
        self.parser.add_argument(
            '--output-dir',
            default="./output",
            help="The folder is for saving results"
        )
        self.parser.add_argument(
            '--exp-name',
            default="exp",
            help="The current experiment name"
        )
        self.parser.add_argument(
            '--test-exp-num',
            default=None,
            help="The current experiment name"
        )
        self.parser.add_argument(
            '--config',
            default='./config/base.yml',
            help='Path to the configuration file'
        )
        
        # Dataset & Transform
        self.parser.add_argument(
            '--dataset',
            default="BaseDataset",
            help="The input dataset type (ex. ProfileDataset)"
        )
        self.parser.add_argument(
            '--augmentation',
            default="BaseAugmentation",
            help="The augmentation method"
        )
        self.parser.add_argument(
            '--resize',
            default=[384, 288],
            help="The input image resize"
        )
        self.parser.add_argument(
            '--valid-ratio',
            default= 0.2,
            help="The dataset split ratio to train/validation"
        )
        self.parser.add_argument(
            '--balanced_split',
            default=True,
            help="The dataset split option"
        )
        self.parser.add_argument(
            '--sampler',
            default="",
            help="The dataloader sampler"
        )
        
        # Training
        self.parser.add_argument(
            '--seed',
            default=777,
            help="The initial random seed"
        )
        self.parser.add_argument(
            '--max-epochs',
            default=10,
            help="The max epochs"
        )
        self.parser.add_argument(
            '--batch-size',
            default=32,
            help="The training batch size"
        )
        self.parser.add_argument(
            '--valid-batch-size',
            default=128,
            help="The validation batch size"
        )
        self.parser.add_argument(
            '--optimizer',
            default="Adam",
            help="The optimizer for training"
        )
        self.parser.add_argument(
            '--lr',
            default=0.0001,
            help="The learning rate for training"
        )
        self.parser.add_argument(
            '--lr-decay-step',
            default=777,
            help="The learning rate decay steps"
        )
        self.parser.add_argument(
            '--criterion',
            default="focal",
            help="The loss function"
        )
        self.parser.add_argument(
            '--log-interval',
            default=10,
            help="The log interval"
        )
        
        # Model
        self.parser.add_argument(
            '--model',
            default="EfficientnetB4",
            help="The model"
        )
        
        # HPO
        self.parser.add_argument(
            '--hpo',
            default=True,
            help="Try HPO"
        )

        self.parser.add_argument(
            '--detection',
            default='False',
            help="Detection in Image"
        )

        self.parser.add_argument(
            '--cutmix',
            default=False,
            help="Cutmix or Mixup in train dataset"
        )
    
    def print_args(self, args):
        print("Arguments:")
        for arg in vars(args):
            print("\t{}: {}".format(arg, getattr(args, arg)))


        