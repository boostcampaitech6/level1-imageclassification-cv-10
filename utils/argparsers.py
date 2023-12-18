import argparse
import os
import yaml

class Parser(object):    
    def __init__(self, description=""):
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)
    
    def create_parser(self):
        self.parser.add_argument(
            '--train-data-dir',
            default="/input",
            help="The dataset folder path"
        )
        
        self.parser.add_argument(
            '--test-data-dir',
            default="/input",
            help="The test dataset folder path"
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
            default="/data/ephemeral/home/project/results/",
            help="The folder is for saving results"
        )
        self.parser.add_argument(
            '--output-dir',
            default="./output",
            help="The folder is for saving results"
        )
        self.parser.add_argument(
            '--project-name',
            default="exp",
            help="The wandb project name"
        )
        self.parser.add_argument(
            '--exp-name',
            default="train",
            help="The current experiment name"
        )
        self.parser.add_argument(
            '--test-exp-num',
            default=None,
            help="The current experiment name"
        )
        self.parser.add_argument(
            '--config',
            default='/data/ephemeral/home/project/config/final_cls_age_rem.yml',
            help='Path to the configuration file'
        )
        
        # Dataset & Transform
        self.parser.add_argument(
            '--dataset',
            default="MaskSplitByProfileDataset",
            help="The input dataset type (ex. ProfileDataset)"
        )
        self.parser.add_argument(
            '--augmentation',
            default="BaseAugmentation",
            help="The augmentation method"
        )
        
        self.parser.add_argument(
            '--resize',
            default=[256, 192],
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
        
        self.parser.add_argument(
            '--age_drop',
            default=False,
            help="ignore 57~59 age data"
        )
        
        # Training
        self.parser.add_argument(
            '--seed',
            default=42,
            help="The initial random seed"
        )
        self.parser.add_argument(
            '--max_epochs',
            default=10,
            help="The max epochs"
        )
        self.parser.add_argument(
            '--batch_size',
            default=128,
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
            default=0.001,
            help="The learning rate for training"
        )
        self.parser.add_argument(
            '--scheduler',
            default="cosine",
            help="The scheduler for training"
        )
        self.parser.add_argument(
            '--lr-decay-step',
            default=100,
            help="The learning rate decay steps"
        )
        self.parser.add_argument(
            '--scheduler',
            default="cosine",
            help="The scheduler for training"
        )
        self.parser.add_argument(
            '--criterion',
            default="cross_entropy",
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
            default="Efficientnet",
            help="The model"
        )
        
        # HPO
        self.parser.add_argument(
            '--hpo',
            default=True,
            help="Try HPO"
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


        